from typing import Iterable

import torch.nn as nn

from data.batch import Batch


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


def model_translations(model: nn.Module, test_data: Iterable, pad_idx=2):
    """TODO: model translations"""
    for b in test_data:
        batch = Batch(b[0], b[1], pad_idx)
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        return {"tgt": batch.tgt, "out": out}
