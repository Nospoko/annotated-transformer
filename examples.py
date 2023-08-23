import hydra
from omegaconf import OmegaConf, DictConfig

from model import make_model
from eval import load_checkpoint
from utils import translate_sample_sentences
from data.dataloaders import load_vocab, load_tokenizers, create_dataloader


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg: DictConfig):
    checkpoint = load_checkpoint(run_name=cfg.run_name, epoch=cfg.model_epoch)
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, train_cfg)

    print("Preparing Data ...")
    test_dataloader = create_dataloader(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        slice="100%",
        split="test",
        batch_size=1,
        shuffle=cfg.random_examples,
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
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checking Model Outputs:")
    translations = translate_sample_sentences(
        valid_dataloader=test_dataloader,
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

    return


if __name__ == "__main__":
    main()
