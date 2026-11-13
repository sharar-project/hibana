# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/op.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl

exp = tl.exp


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))
