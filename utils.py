from typing import Iterable

import torch
import torchtext
import torch.nn as nn

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


def translate_sample_sentences(
    dataloader: Iterable,
    model: nn.Module,
    vocab_src: torchtext.vocab.Vocab,
    vocab_tgt: torchtext.vocab.Vocab,
    pad_idx: int = 2,
    n_examples: int = 20,
    eos_string: str = "</s>",
):
    src_itos = vocab_src.get_itos()
    tgt_itos = vocab_tgt.get_itos()

    results = []
    for batch in dataloader:
        for it in range(len(batch)):
            record = batch[it]

            src_tokens = [src_itos[x] for x in record.src if x != pad_idx]
            tgt_tokens = [tgt_itos[x] for x in record.tgt if x != pad_idx]

            decoded_record = greedy_decode(
                model=model,
                src=record.src,
                src_mask=record.src_mask,
                max_len=72,
                start_symbol=0,
            )

            model_txt = [tgt_itos[x] for x in decoded_record if x != pad_idx]

            result = {
                "src": " ".join(src_tokens).replace("\n", ""),
                "tgt": " ".join(tgt_tokens).replace("\n", ""),
                "out": " ".join(model_txt).split(eos_string, 1)[0] + eos_string,
            }
            results.append(result)

            if len(results) == n_examples:
                return results

    return results


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
) -> torch.Tensor:
    # Pretend to be batches
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)

    memory = model.encode(src, src_mask)
    # Create a tensor and put start symbol inside
    sentence = torch.Tensor([[start_symbol]]).type_as(src.data)
    for _ in range(max_len - 1):
        sub_mask = subsequent_mask(sentence.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, sentence, sub_mask)

        prob = model.generator(out[:, -1])
        _, next_word = prob.max(dim=1)
        next_word = next_word.data[0]

        sentence = torch.cat([sentence, torch.Tensor([[next_word]]).type_as(src.data)], dim=1)

    # Don't pretend to be a batch
    return sentence[0]
