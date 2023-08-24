from os.path import exists
from typing import Callable, Iterable

import spacy
import torch
import torchtext.vocab
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator

from data.batch import Batch
from data.dataset import TranslationDataset
from data.tokenization import tokenize, yield_tokens, load_tokenizers


def build_vocabulary(
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    max_tokens: int,
) -> tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab]:
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    dataset = TranslationDataset(language_pair="de-en", split="train")
    print("Building German Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(dataset, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        max_tokens=max_tokens,
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(dataset, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        max_tokens=max_tokens,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    max_tokens: int,
) -> tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab]:
    vocab_path = f"vocab-{max_tokens}.pt"
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en, max_tokens)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path)

    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


# code for evenly dividing torchtext data to batches
def collate_batch(
    batch: Iterable,
    src_tokens_fn: Callable,
    tgt_tokens_fn: Callable,
    src_vocab: torchtext.vocab.Vocab,
    tgt_vocab: torchtext.vocab.Vocab,
    device=torch.device("cpu"),
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [bs_id, torch.tensor(src_vocab(src_tokens_fn(_src)), dtype=torch.int64, device=device), eos_id],
            0,
        )
        processed_tgt = torch.cat(
            [bs_id, torch.tensor(tgt_vocab(tgt_tokens_fn(_tgt)), dtype=torch.int64, device=device), eos_id],
            0,
        )
        # warning - overwrites values for negative values of padding - len
        src_list.append(
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return Batch(src, tgt, pad=pad_id)


def create_dataloaders(
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    slice: str,
    device=torch.device("cpu"),
    batch_size=12000,
    max_padding=128,
    split_val="validation",
):
    train_dataloader = create_dataloader(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice,
        device,
        batch_size,
        max_padding,
        split="train",
        shuffle=True,
    )
    valid_dataloader = create_dataloader(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        slice,
        device,
        batch_size,
        max_padding,
        split=split_val,
    )

    return train_dataloader, valid_dataloader


def create_dataloader(
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    slice: str,
    device=torch.device("cpu"),
    batch_size=12000,
    max_padding=128,
    split="train",
    shuffle=False,
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
            device=device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    data = TranslationDataset(split=f"{split}[:{slice}]", language_pair="de-en")

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


if __name__ == "__main__":

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    dataset = TranslationDataset(split="validation[:5]", language_pair="de-en")
    spacy_de, spacy_en = load_tokenizers()
    # FIXME
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, "1%")
    train_data, vel_data = create_dataloaders(vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=16, slice="5")

    index = torch.randint(0, len(dataset), [1])
    print(index)

    src = dataset[index][0]
    tgt = dataset[index][1]
    bs_id = torch.tensor([0])  # <s> token id
    eos_id = torch.tensor([1])  # </s> token id

    tokenized = tokenize_de(src)
    coded = vocab_src(tokenized)

    processed_src = torch.cat(
        [bs_id, torch.tensor(coded, dtype=torch.int64), eos_id],
        0,
    )

    print(f"Source:\n{src}\ntokenized: {tokenized}\nencoded in vocabulary: {coded}\ntensor: {processed_src}")

    tokenized = tokenize_en(tgt)
    coded = vocab_tgt(tokenized)

    processed_tgt = torch.cat(
        [bs_id, torch.tensor(coded, dtype=torch.int64), eos_id],
        0,
    )
    print(f"Target:\n{tgt}\ntokenized: {tokenized}\nencoded in vocabulary: {coded}\ntensor: {processed_src}")
