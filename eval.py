import hydra
import os
from model import make_model
import torch
from train import val_epoch
from modules.label_smoothing import LabelSmoothing
from data.dataloaders import load_vocab, load_tokenizers, create_dataloaders


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg):
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
    checkpoint = load_checkpoint(run_id=cfg.run_id)
    model_cfg = checkpoint['cfg']

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        n=model_cfg['model']['n'],
        d_model=model_cfg['model']['d_model'],
        d_ff=model_cfg['model']['d_ff'],
        dropout=model_cfg['model']['dropout'],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    pad_idx = vocab_tgt["<blank>"]
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)

    loss = val_epoch(data_iter=test_dataloader, model=model, criterion=criterion)
    print(f"Model loss:   {loss}")


def load_checkpoint(run_id: str):
    # find path with desired run_id
    path = None
    for file in os.listdir("models"):
        if file.find(f"{run_id}-final.pt"):
            path = file
        break
    if path is None:
        print("no run with this id found")
        return None
    path = "models/" + path
    checkpoint = torch.load(path)
    return checkpoint


if __name__ == '__main__':
    main()
