from datasets import load_dataset


class TranslationDataset:
    def __init__(self, split="train", language_pair="de-en"):
        self.language_pair = language_pair
        self.dataset = load_dataset("wmt16", name=language_pair, split=split)
        self._build()

    def _build(self):
        self.samples = []
        for row in self.dataset:
            self.samples.append(tuple(list(row.values())[0].values()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
