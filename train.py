import time
import uuid
from typing import Callable, Iterable

import hydra
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm
import torchtext.vocab.vocab
from torch.optim.lr_scheduler import LambdaLR
from omegaconf.omegaconf import OmegaConf, DictConfig

import wandb
from data.batch import Batch
from model import make_model
from modules.label_smoothing import LabelSmoothing
from utils import TrainState, SimpleLossCompute, rate
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    # load tokenizers and vocab
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cfg.data_slice)

    # Train a model
    model, run_id = train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, cfg)

    # save weights to a file
    file_path = f"models/{cfg.file_prefix}-{run_id}-final.pt"
    torch.save(model.state_dict(), file_path)

    # print a run_id of the model
    print(run_id)


def initialize_wandb(cfg: DictConfig) -> str:
    # initialize experiment on WandB with unique run id
    run_id = str(uuid.uuid1())[:8]
    wandb.init(
        project=cfg.project,
        name=f"{cfg.run}-{run_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run_id


def train_model(
    vocab_src: torchtext.vocab.Vocab,
    vocab_tgt: torchtext.vocab.Vocab,
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    cfg: DictConfig,
) -> tuple[nn.Module, str]:
    run_id = initialize_wandb(cfg)

    # Get the index for padding token
    pad_idx = vocab_tgt["<blank>"]

    # define model parameters and create the model
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    module = model

    # Set LabelSmoothing as a criterion for loss calculation
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)

    # Create dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice=cfg.data_slice,
        batch_size=cfg.batch_size,
        max_padding=cfg.max_padding,
    )

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=cfg.warmup),
    )

    train_state = TrainState()

    for epoch in range(cfg.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, train_state = train_epoch(
            train_dataloader,
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            pad_idx,
            accum_iter=cfg["accum_iter"],
            train_state=train_state,
        )

        # Save checkpoint after each epoch
        file_path = "models/%s-%s-%.2d.pt" % (cfg.file_prefix, run_id, epoch)
        torch.save(module.state_dict(), file_path)

        print(f"Epoch {epoch} Validation", flush=True)
        model.eval()
        # Evaluate the model on validation set
        sloss, val_state = val_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
        )
        # Log validation and training losses
        print(sloss)
        wandb.log({"val/loss": sloss, "train/loss": t_loss})

    return model, run_id


def train_epoch(
    data_iter: Iterable,
    model: nn.Module,
    loss_compute: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    pad_idx: int,
    accum_iter=1,
    train_state=TrainState(),
) -> tuple[float, TrainState]:
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    i = 0  # iteration counter

    # create progress bar
    pbar = tqdm(data_iter)
    for b in pbar:  # for batch in dataloader
        batch = Batch(b[0], b[1], pad_idx)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss_node.backward()

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        i += 1

        # Update learning rate scheduler
        scheduler.step()

        # Update loss and token counts
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # Display progress every 2 iterations
        if i % 2 == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            loss = loss / batch.ntokens
            tok_rate = tokens / elapsed
            pbar.set_description(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss:6.2f} | Tokens / Sec {tok_rate:7.1f} | Learning Rate: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            wandb.log({"train_steps/loss": loss})
        del loss
        del loss_node
    # Return average loss over all tokens and updated train state
    return total_loss / total_tokens, train_state


def val_epoch(
    data_iter: Iterable,
    model: nn.Module,
    loss_compute: Callable,
    train_state=TrainState(),
) -> tuple[float, TrainState]:
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        del loss
        del loss_node
    # Return average loss over all tokens and updated train state
    return total_loss / total_tokens, train_state


if __name__ == "__main__":
    main()
