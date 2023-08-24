import time
from typing import Callable, Iterable

import hydra
import spacy
import torch
import einops
import torch.nn as nn
from tqdm import tqdm
import torchtext.vocab.vocab
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import LambdaLR

import wandb
from model import make_model
from utils import TrainState, rate
from modules.label_smoothing import LabelSmoothing
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    # load tokenizers and vocab
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cfg)
    vocab_src_size = len(vocab_src)
    vocab_tgt_size = len(vocab_tgt)

    # Train a model
    initialize_wandb(cfg)
    print(f"Train using {cfg.device}")

    model = train_model(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        cfg=cfg,
        device=cfg.device,
    )

    # save weights to a file
    file_path = f"models/{cfg.file_prefix}-{cfg.run_name}-final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "vocab_src_size": vocab_src_size,
            "vocab_tgt_size": vocab_tgt_size,
        },
        file_path,
    )


def initialize_wandb(cfg: DictConfig) -> str:
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def train_model(
    vocab_src: torchtext.vocab.Vocab,
    vocab_tgt: torchtext.vocab.Vocab,
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    cfg: DictConfig,
    device: str = "cpu",
) -> nn.Module:
    # Get the index for padding token
    pad_idx = vocab_tgt["<blank>"]
    vocab_src_size = len(vocab_src)
    vocab_tgt_size = len(vocab_tgt)

    # define model parameters and create the model
    model = make_model(
        vocab_src_size=vocab_src_size,
        vocab_tgt_size=vocab_tgt_size,
        n=cfg.model.n,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        h=cfg.model.h,
        dropout=cfg.model.dropout,
    )
    model.to(device)

    # Set LabelSmoothing as a criterion for loss calculation
    criterion = LabelSmoothing(
        size=vocab_tgt_size,
        padding_idx=pad_idx,
        smoothing=cfg.train.label_smoothing,
    )
    criterion.to(device)

    # Create dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        device=device,
        slice=cfg.data_slice,
        max_padding=cfg.max_padding,
        batch_size=cfg.train.batch_size,
    )

    # Define optimizer and learning rate lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, cfg.model.d_model, factor=1, warmup=cfg.warmup),
    )
    train_state = TrainState()
    for epoch in range(cfg.train.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, train_state = train_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_state=train_state,
            accum_iter=cfg.train.accum_iter,
            log_frequency=cfg.log_frequency,
        )

        # Save checkpoint after each epoch
        file_path = f"models/{cfg.file_prefix}-{cfg.run_name}-{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "vocab_src_size": len(vocab_src),
                "vocab_tgt_size": len(vocab_tgt),
            },
            file_path,
        )

        print(f"Epoch {epoch} Validation", flush=True)
        with torch.no_grad():
            model.eval()
            # Evaluate the model on validation set
            v_loss = val_epoch(
                dataloader=valid_dataloader,
                model=model,
                criterion=criterion,
            )

        # Log validation and training losses
        wandb.log({"val/loss_epoch": v_loss, "train/loss_epoch": t_loss})
    return model


def train_epoch(
    dataloader: Iterable,
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    train_state: TrainState,
    accum_iter: int = 1,
    log_frequency: int = 10,
) -> tuple[float, TrainState]:
    start = time.time()
    total_loss = 0
    tokens = 0
    n_accum = 0
    it = 0

    # create progress bar
    pbar = tqdm(dataloader)
    steps = len(dataloader)
    for batch in pbar:
        encode_decode = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encode_decode)

        out = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out, target) / batch.ntokens
        loss.backward()

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if it % accum_iter == 0 or it == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        it += 1

        # Update learning rate lr_scheduler
        lr_scheduler.step()

        # Update loss and token counts
        loss_item = loss.item()
        total_loss += loss.item()
        tokens += batch.ntokens

        # log metrics every log_frequency steps
        if it % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            pbar.set_description(
                f"Step: {it:6d}/{steps} | acc_step: {n_accum:3d} | Loss: {loss_item:6.2f}"
                + f"| tps: {tok_rate:7.1f} | LR: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            wandb.log({"train/loss_step": loss.item()})

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), train_state


@torch.no_grad()
def val_epoch(
    dataloader: Iterable,
    model: nn.Module,
    criterion: Callable,
) -> float:
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for batch in tqdm(dataloader):
        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        tgt_rearranged = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out_rearranged, tgt_rearranged) / batch.ntokens

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader)


if __name__ == "__main__":
    main()
