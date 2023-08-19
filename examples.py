import os
from typing import Iterable

import hydra
import torch
import torch.nn as nn
import torchtext.vocab.vocab
from omegaconf import DictConfig

from data.batch import Batch
from model import make_model
from modules.encoderdecoder import subsequent_mask
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
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checking Model Outputs:")
    check_outputs(test_dataloader, model, vocab_src, vocab_tgt, n_examples=cfg.n_examples)
    return


def check_outputs(
    valid_dataloader: Iterable,
    model: nn.Module,
    vocab_src: torchtext.vocab.Vocab,
    vocab_tgt: torchtext.vocab.Vocab,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        batch = Batch(b[0], b[1], pad_idx)

        src_tokens = [vocab_src.get_itos()[x] for x in batch.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in batch.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        model_out = greedy_decode(model, batch.src, batch.src_mask, 72, 0)[0]
        model_txt = [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        sentences_txt = " ".join(model_txt).split(eos_string, 1)[0] + eos_string
        print("Model Output               : " + sentences_txt.replace("\n", ""))
        results[idx] = (batch, src_tokens, tgt_tokens, model_out, model_txt)

    return results


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # Create a tensor and put start symbol inside
    sentence = torch.Tensor([[start_symbol]]).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, sentence, subsequent_mask(sentence.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = prob.max(dim=1)
        next_word = next_word.data[0]
        sentence = torch.cat([sentence, torch.Tensor([[next_word]]).type_as(src.data)], dim=1)
    return sentence


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
