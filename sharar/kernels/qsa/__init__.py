# SPDX-License-Identifier: Apache-2.0
"""QSA kernels, with lazy GPU imports for CPU-only control and packaging tools."""

__all__ = ["sparse_attention"]


def __getattr__(name):
    if name == "sparse_attention":
        from .attention import sparse_attention

        return sparse_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
