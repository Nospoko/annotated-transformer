import time
import uuid
from typing import Callable, Iterable

import hydra
import spacy
import torch
import einops
import torch.nn as nn
from tqdm import tqdm
import torchtext.vocab.vocab
from torch.optim.lr_scheduler import LambdaLR
from omegaconf.omegaconf import OmegaConf, DictConfig

import wandb
from data.batch import Batch
from model import make_model
from utils import TrainState, rate
from modules.label_smoothing import LabelSmoothing
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    # load tokenizers and vocab
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cfg.data_slice)
    vocab_src_size = len(vocab_src)
    vocab_tgt_size = len(vocab_tgt)
    # Train a model
    model, run_id = train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, cfg, device=cfg.device)

    # save weights to a file
    file_path = f"models/{cfg.file_prefix}-{run_id}-final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "vocab_src_size": vocab_src_size,
            "vocab_tgt_size": vocab_tgt_size,
        },
        file_path,
    )

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
    device="cpu",
) -> tuple[nn.Module, str]:
    print(f"Train using {device}")

    run_id = initialize_wandb(cfg)

    # Get the index for padding token
    pad_idx = vocab_tgt["<blank>"]
    vocab_src_size = len(vocab_src)
    vocab_tgt_size = len(vocab_tgt)

    # define model parameters and create the model
    model = make_model(
        vocab_src_size,
        vocab_tgt_size,
        n=cfg.model.n,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        h=cfg.model.h,
        dropout=cfg.model.dropout,
    )
    model.to(device)

    # Set LabelSmoothing as a criterion for loss calculation
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    # Create dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        device=device,
        slice=cfg.data_slice,
        batch_size=cfg.batch_size,
        max_padding=cfg.max_padding,
    )

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, cfg.model.d_model, factor=1, warmup=cfg.warmup),
    )

    train_state = TrainState()

    for epoch in range(cfg.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, train_state = train_epoch(
            train_dataloader,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            pad_idx=pad_idx,
            train_state=train_state,
            accum_iter=cfg["accum_iter"],
            log_frequency=cfg.log_frequency,
        )

        # Save checkpoint after each epoch
        file_path = f"models/{cfg.file_prefix}-{run_id}-{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "vocab_src_size": len(vocab_src),
                "vocab_tgt_size": len(vocab_tgt),
            },
            file_path,
        )

        torch.cuda.empty_cache()

        print(f"Epoch {epoch} Validation", flush=True)
        model.eval()
        # Evaluate the model on validation set
        sloss = val_epoch(
            train_dataloader,
            model,
            criterion,
        )
        # Log validation and training losses
        print(sloss)
        wandb.log({"val/loss": sloss, "train/loss": t_loss})

        torch.cuda.empty_cache()

    return model, run_id


def train_epoch(
    data_iter: Iterable,
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    train_state: TrainState,
    pad_idx=2,
    accum_iter=1,
    log_frequency=10,
) -> tuple[float, TrainState]:
    start = time.time()
    total_loss = 0
    tokens = 0
    n_accum = 0
    i = 0  # iteration counter

    # create progress bar
    pbar = tqdm(data_iter)

    for b in pbar:  # for batch in dataloader
        batch = Batch(b[0], b[1], pad_idx)

        encode_decode = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encode_decode)

        loss = criterion(einops.rearrange(out, "b n d -> (b n) d"), einops.rearrange(batch.tgt_y, "b n -> (b n)")) / batch.ntokens
        loss.backward()

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
        total_loss += loss.item()
        tokens += batch.ntokens

        # log metrics every 10 steps
        if i % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            pbar.set_description(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss.item():6.2f} | Tokens / Sec {tok_rate:7.1f} | Learning Rate: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            wandb.log({"train_steps/loss": loss.item()})
    del optimizer
    del lr
    # Return average loss over all tokens and updated train state
    return total_loss / len(data_iter), train_state


def val_epoch(
    data_iter: Iterable,
    model: nn.Module,
    criterion: Callable,
    pad_idx=2,
) -> float:
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for b in tqdm(data_iter):
        batch = Batch(b[0], b[1], pad_idx)
        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)
        loss = criterion(einops.rearrange(out, "b n d -> (b n) d"), einops.rearrange(batch.tgt_y, "b n -> (b n)")) / batch.ntokens

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

    # Return average loss over all tokens and updated train state
    return total_loss / len(data_iter)


if __name__ == "__main__":
    main()
