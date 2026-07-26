# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Weight-free KV cache sizing helpers for the ``sim`` connector.

Pure-python (no torch / no CUDA) so it works on a GPU-free host.  Shared by:

- the Phase 1 transfer-rate knob (worker derives per-request transferred bytes
  from its allocated block count to pace ``get_finished``), and
- the Phase 2 GPU-free gating in ``ModelRunner`` (``get_num_blocks`` /
  ``_compute_block_bytes``).

Everything is derived from the HuggingFace config, so no weights are needed.
"""

from __future__ import annotations

from typing import Any

# Map common KV-cache dtype spellings to bytes-per-element.  Kept as plain
# strings so this module never imports torch (must stay GPU-free).
_DTYPE_ITEMSIZE: dict[str, int] = {
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "float8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "float": 4,
}


def kv_dtype_itemsize(kv_dtype: Any) -> int:
    """Return bytes-per-element for a KV cache dtype.

    Accepts a string (``"fp8"``, ``"bf16"``, ...), a torch/numpy dtype (via its
    ``str``), or ``None`` / ``"auto"`` (defaults to 2 bytes = fp16/bf16).
    Unknown spellings fall back to 2 bytes.
    """
    if kv_dtype is None:
        return 2
    name = str(kv_dtype).lower()
    if name in ("auto", ""):
        return 2
    # Normalise things like "torch.bfloat16" / "torch.float8_e4m3fn".
    name = name.rsplit(".", 1)[-1]
    return _DTYPE_ITEMSIZE.get(name, 2)


def _hf_attr(hf_config: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute among *names* (HF configs vary)."""
    for name in names:
        val = getattr(hf_config, name, None)
        if val is not None:
            return val
    return default


def sim_block_bytes(
    hf_config: Any,
    block_size: int,
    kv_dtype: Any = None,
    tp_size: int = 1,
) -> int:
    """Estimate the per-rank byte size of one KV cache block.

    A block holds ``block_size`` tokens of K and V for every layer and every
    KV head owned by this tensor-parallel rank::

        block_bytes = 2(K+V) * num_layers * block_size
                      * kv_heads_per_rank * head_dim * itemsize

    This is an approximation for transfer-rate pacing and GPU-free block
    accounting; it deliberately ignores fp8 scale side-cars and any padding.

    Args:
        hf_config: HuggingFace model config (attributes only, no weights).
        block_size: Tokens per KV block.
        kv_dtype: KV cache dtype (string or torch/numpy dtype); defaults 2 bytes.
        tp_size: Tensor-parallel world size; KV heads are sharded across ranks.

    Returns:
        Bytes for one KV block on a single rank (always ``>= 1``).
    """
    num_layers = int(_hf_attr(hf_config, "num_hidden_layers", "n_layer", default=1))
    num_attn_heads = int(
        _hf_attr(hf_config, "num_attention_heads", "n_head", default=1)
    )
    hidden_size = int(
        _hf_attr(hf_config, "hidden_size", "n_embd", default=num_attn_heads)
    )
    num_kv_heads = int(
        _hf_attr(
            hf_config,
            "num_key_value_heads",
            "num_kv_heads",
            default=num_attn_heads,
        )
    )
    head_dim = int(
        _hf_attr(hf_config, "head_dim", default=max(1, hidden_size // num_attn_heads))
    )

    kv_heads_per_rank = max(1, num_kv_heads // max(1, tp_size))
    itemsize = kv_dtype_itemsize(kv_dtype)

    block_bytes = (
        2  # K and V
        * num_layers
        * int(block_size)
        * kv_heads_per_rank
        * head_dim
        * itemsize
    )
    return max(1, int(block_bytes))
