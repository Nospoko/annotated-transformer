import time
import uuid
from os.path import exists
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
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    module, run_id = load_trained_model(cfg)
    file_path = f"models/{cfg.file_prefix}-{run_id}-final.pt"
    torch.save(module.state_dict(), file_path)
    print(run_id)


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def load_trained_model(cfg) -> tuple[nn.Module, str]:
    # load tokenizers and vocab
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cfg.data_slice)
    # if model from cfg does not exist - train a new model
    if not exists(cfg.model_path):
        model, run_id = train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, cfg)
    else:
        model = make_model(len(vocab_src), len(vocab_tgt), n=6)
        model.load_state_dict(torch.load(cfg.model_path))
    return model, run_id


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
    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    module = model
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    train_dataloader, valid_dataloader = create_dataloaders(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice=cfg.data_slice,
        batch_size=cfg.batch_size,
        max_padding=cfg.max_padding,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=cfg.warmup),
    )
    train_state = TrainState()

    for epoch in range(cfg.num_epochs):
        model.train()
        print(f"Epoch {epoch} Training ====", flush=True)
        t_loss, train_state = train_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            accum_iter=cfg["accum_iter"],
            train_state=train_state,
        )

        file_path = "models/%s-%s-%.2d.pt" % (cfg.file_prefix, run_id, epoch)
        torch.save(module.state_dict(), file_path)

        print(f"Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss, val_state = val_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
        )
        print(sloss)
        wandb.log({"val/loss": sloss, "train/loss": t_loss})

    return model, run_id


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


def train_epoch(
    data_iter: Iterable,
    model: nn.Module,
    loss_compute: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accum_iter=1,
    train_state=TrainState(),
) -> tuple[float, TrainState]:
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    pbar = tqdm(enumerate(data_iter))
    for i, batch in pbar:
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss_node.backward()
        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()
    total_loss += loss
    total_tokens += batch.ntokens
    tokens += batch.ntokens
    if i % 40 == 1:
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        loss = loss / batch.ntokens
        tok_rate = tokens / elapsed
        pbar.set_description(
            f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss:6.2f} | Tokens / Sec {tok_rate:7.1f} | Learning Rate: {lr:6.1e}"
        )
        wandb.log({"train_steps/loss": loss / batch.ntokens})
        del loss
        del loss_node
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
    return total_loss / total_tokens, train_state


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


if __name__ == "__main__":
    main()
