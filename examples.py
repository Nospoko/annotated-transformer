import hydra
from omegaconf import DictConfig

from model import make_model
from eval import load_checkpoint
from utils import translated_sentences
from data.dataloaders import load_vocab, load_tokenizers, create_dataloader


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg: DictConfig):
    checkpoint = load_checkpoint(run_id=cfg.run_id, epoch=cfg.model_epoch)
    model_cfg = checkpoint["cfg"]
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, model_cfg["vocab_data_slice"])

    print("Preparing Data ...")
    test_dataloader = create_dataloader(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice=model_cfg["data_slice"],
        split="test",
        batch_size=1,
        shuffle=cfg.random_examples,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checking Model Outputs:")
    translations = translated_sentences(test_dataloader, model, vocab_src, vocab_tgt, n_examples=cfg.n_examples)
    for idx in range(cfg.n_examples):
        print("Source Text (Input)        : " + translations[idx]["src"])
        print("Target Text (Ground Truth) : " + translations[idx]["tgt"])
        print("Model Output               : " + translations[idx]["out"])
        print("===========")

    return


if __name__ == "__main__":
    main()
