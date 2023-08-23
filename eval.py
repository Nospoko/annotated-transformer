import os

import hydra
import torch

from train import val_epoch
from model import make_model
from modules.label_smoothing import LabelSmoothing
from data.dataloaders import load_vocab, load_tokenizers, create_dataloader


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg):
    checkpoint = load_checkpoint(run_id=cfg.run_id, device=cfg.device)
    model_cfg = checkpoint["cfg"]
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, model_cfg["data_slice"])
    print("Preparing Data ...")
    test_dataloader = create_dataloader(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice=model_cfg["data_slice"],
        split="test",
        device=cfg.device,
        batch_size=1,
    )

    print("Loading Trained Model ...")

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        n=model_cfg["model"]["n"],
        d_model=model_cfg["model"]["d_model"],
        d_ff=model_cfg["model"]["d_ff"],
        dropout=model_cfg["model"]["dropout"],
    )
    model.to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pad_idx = vocab_tgt["<blank>"]
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(cfg.device)
    print("Evaluating model ...")
    loss = val_epoch(data_iter=test_dataloader, model=model, criterion=criterion)
    print(f"Model loss:   {loss}")


def load_checkpoint(run_id: str, epoch="final", device="cpu"):
    # find path with desired run_id
    path = None
    for file in os.listdir("models"):
        if file.find(f"{run_id}-{epoch}") > 0:
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
