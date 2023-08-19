from os.path import exists

import spacy
import torch
import torchtext.vocab
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator

from data.dataset import TranslationDataset
from data.tokenization import tokenize, yield_tokens, load_tokenizers


def build_vocabulary(
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    slice: str,
) -> tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab]:
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


def load_vocab(
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    slice: str,
) -> tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab]:
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en, slice)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


# code for evenly dividing torchtext data to batches
def collate_batch(
    batch,
    src_tokens_fn,
    tgt_tokens_fn,
    src_vocab,
    tgt_vocab,
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
    return (src, tgt)


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

    train_data = TranslationDataset(split=f"train[:{slice}]", language_pair="de-en")
    val_data = TranslationDataset(split=f"{split_val}[:{slice}]", language_pair="de-en")

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

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    dataset = TranslationDataset(split="validation[:5]", language_pair="de-en")
    spacy_de, spacy_en = load_tokenizers()
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
