import os

import spacy

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


if __name__ == "__main__":
    dataset = TranslationDataset(split="validation[:5]", language_pair="de-en")
    spacy_de, _ = load_tokenizers()
    print(tokenize(dataset[0][0], spacy_de))
