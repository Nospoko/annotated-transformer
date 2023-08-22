from typing import Iterable

import torch
import torchtext
import torch.nn as nn
from tqdm import tqdm

from data.batch import Batch
from modules.encoderdecoder import subsequent_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def translated_sentences(
    valid_dataloader: Iterable,
    model: nn.Module,
    vocab_src: torchtext.vocab.Vocab,
    vocab_tgt: torchtext.vocab.Vocab,
    pad_idx=2,
    n_examples=20,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in tqdm(range(n_examples)):
        b = next(iter(valid_dataloader))
        batch = Batch(b[0], b[1], pad_idx)
        src_tokens = [vocab_src.get_itos()[x] for x in batch.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in batch.tgt[0] if x != pad_idx]

        model_out = greedy_decode(model, batch.src, batch.src_mask, 72, 0)[0]
        model_txt = [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        results[idx] = {
            "src": " ".join(src_tokens).replace("\n", ""),
            "tgt": " ".join(tgt_tokens).replace("\n", ""),
            "out": " ".join(model_txt).split(eos_string, 1)[0] + eos_string,
        }
    return results


def greedy_decode(model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int):
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
