import os

import hydra
import torch
from omegaconf import DictConfig

from model import make_model
from utils import translated_sentences
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg: DictConfig):
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cfg.data_slice)
    print("Preparing Data ...")
    train_dataloader, test_dataloader = create_dataloaders(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice=cfg.data_slice,
        split_val="test",
        batch_size=1,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    checkpoint = load_checkpoint(run_id=cfg.run_id)
    model.load_state_dict(checkpoint)  # ["model_state_dict"])
    print("Checking Model Outputs:")
    translations = translated_sentences(test_dataloader, model, vocab_src, vocab_tgt, n_examples=cfg.n_examples)
    for idx in range(cfg.n_examples):
        print("Source Text (Input)        : " + translations[idx]["src"])
        print("Target Text (Ground Truth) : " + translations[idx]["tgt"])
        print("Model Output               : " + translations[idx]["out"])
        print("===========")

    return


def load_checkpoint(run_id: str):
    # find path with desired run_id
    path = None
    for file in os.listdir("models"):
        if file.find(f"{run_id}-06.pt"):
            path = file
        break
    if path is None:
        print("no run with this id found")
        return None
    path = "models/" + path
    checkpoint = torch.load(path)
    return checkpoint


if __name__ == "__main__":
    main()
