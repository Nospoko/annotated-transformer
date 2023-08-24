import os

import hydra
import torch
from omegaconf import OmegaConf

from train import val_epoch
from model import make_model
from utils import translate_sample_sentences
from modules.label_smoothing import LabelSmoothing
from data.dataloaders import load_vocab, load_tokenizers, create_dataloader


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg):
    checkpoint = load_checkpoint(
        run_name=cfg.run_name,
        epoch=cfg.model_epoch,
        device=cfg.device,
    )
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, train_cfg.max_tokens)

    print("Preparing Data ...")
    test_dataloader = create_dataloader(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        slice="100%",
        split="test",
        device=cfg.device,
        batch_size=train_cfg.train.batch_size,
    )

    print("Loading Trained Model ...")
    model = make_model(
        vocab_src_size=len(vocab_src),
        vocab_tgt_size=len(vocab_tgt),
        n=train_cfg["model"]["n"],
        d_model=train_cfg["model"]["d_model"],
        d_ff=train_cfg["model"]["d_ff"],
        dropout=train_cfg["model"]["dropout"],
    )
    model.to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if cfg.tasks.val_epoch:
        pad_idx = vocab_tgt["<blank>"]
        criterion = LabelSmoothing(
            size=len(vocab_tgt),
            padding_idx=pad_idx,
            smoothing=train_cfg.train.label_smoothing,
        )
        criterion.to(cfg.device)

        print("Evaluating model ...")
        loss = val_epoch(
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
        )
        print(f"Model loss:   {loss}")

    if cfg.tasks.translation:
        print("Checking Model Outputs:")
        translations = translate_sample_sentences(
            dataloader=test_dataloader,
            model=model,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            n_examples=cfg.n_examples,
        )

        for translation in translations:
            print("Source Text (Input)        : " + translation["src"])
            print("Target Text (Ground Truth) : " + translation["tgt"])
            print("Model Output               : " + translation["out"])
            print("===========")


def load_checkpoint(run_name: str, epoch: str = "final", device: str = "cpu"):
    # find path with desired run
    path = None
    for file in os.listdir("models"):
        if file.find(f"{run_name}-{epoch}") > 0:
            path = file
            break

    if path is None:
        print("no run with this id found")
        return None

    path = "models/" + path
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


if __name__ == "__main__":
    main()
