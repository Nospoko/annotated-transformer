import os
from os.path import exists

import spacy
import torch
from torch.utils.data import DataLoader
from torchtext.vocab.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator

from data.batch import collate_batch
from data.dataset import TranslationDataset


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to in data_iter:
        yield tokenizer(from_to[index])


def build_vocabulary(spacy_de, spacy_en, slice):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    dataset = TranslationDataset(language_pair="de-en", split=f"train[:{slice}]+test[:{slice}]+validation[:{slice}]")
    print("Building German Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(dataset, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(dataset, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de: spacy.Language, spacy_en: spacy.Language, slice: str):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en, slice)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def create_dataloaders(
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de,
    spacy_en,
    slice: str,
    batch_size=12000,
    max_padding=128,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_data = TranslationDataset(split=f"train[:{slice}]", language_pair="de-en")
    val_data = TranslationDataset(split=f"validation[:{slice}]", language_pair="de-en")

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    dataset = TranslationDataset(split="validation[:1%]", language_pair="de-en")
    # print(dataset[4].values())
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, "1%")
    print("end")
