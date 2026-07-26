# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for GPU-free simulator gating in ModelRunner (Phase 2).

Validates that with ``config.simulator=True`` and ``torch.cuda`` mocked off,
ModelRunner's sim paths produce correct results without a real GPU:

- ``_compute_block_bytes`` returns a positive integer via ``sim_block_bytes``.
- ``_get_num_blocks_sim`` returns a dict with positive ``num_kvcache_blocks``.
- ``_run_model_sim`` returns tensors of the expected shapes without
  calling ``self.model``.

Mirrors the GPU-free mocking style of ``test_sim_decode_connector.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stub heavy GPU/AITER dependencies before any atom.model_engine import.
# conftest.py already stubs atom.config and atom.utils.forward_context but
# model_runner.py also pulls in aiter, mori, and several atom.* submodules
# that are unavailable in a GPU-free test environment.
# ---------------------------------------------------------------------------
import sys
import types as _types
from unittest.mock import MagicMock

# aiter and its submodules
for _name in [
    "aiter",
    "aiter.dist",
    "aiter.dist.parallel_state",
    "aiter.dist.utils",
]:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# atom submodules that model_runner imports at module level
_stub_modules = [
    "atom.distributed",
    "atom.distributed.pcp_utils",
    "atom.kv_transfer",
    "atom.kv_transfer.disaggregation",
    "atom.model_loader",
    "atom.model_loader.loader",
    "atom.model_ops",
    "atom.model_ops.eplb",
    "atom.model_ops.rejection_sampler",
    "atom.model_ops.sampler",
    "atom.spec_decode",
    "atom.spec_decode.eagle",
    "atom.utils.cuda_graph",
    "atom.utils.selector",
    "atom.utils.tbo",
]
for _name in _stub_modules:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# atom.utils is already a real package but we need some attributes
_atom_utils = sys.modules.get("atom.utils")
if _atom_utils is None:
    _atom_utils = MagicMock()
    sys.modules["atom.utils"] = _atom_utils
# Ensure the attributes model_runner imports from atom.utils exist
for _attr in [
    "CpuGpuBuffer",
    "envs",
    "get_hf_text_config",
    "init_exit_handler",
    "resolve_obj_by_qualname",
]:
    if not hasattr(_atom_utils, _attr):
        setattr(_atom_utils, _attr, MagicMock())

# Patch forward_context to provide set_kv_cache_data (needed by
# _allocate_kv_cache_sim) on top of the conftest stub.
_fc = sys.modules.get("atom.utils.forward_context")
if _fc is not None:
    for _attr in [
        "Context",
        "DPMetadata",
        "ForwardMode",
        "get_forward_context",
        "reset_forward_context",
        "set_forward_context",
        "set_kv_cache_data",
    ]:
        if not hasattr(_fc, _attr):
            setattr(_fc, _attr, MagicMock())
else:
    sys.modules["atom.utils.forward_context"] = MagicMock()

# model_engine sub-imports
for _name in [
    "atom.model_engine.run_labels",
]:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# Ensure CUDAGraphMode and set_current_atom_config exist on the config stub
_atom_config = sys.modules.get("atom.config")
if _atom_config is not None:
    if not hasattr(_atom_config, "CUDAGraphMode"):
        _atom_config.CUDAGraphMode = MagicMock()
    if not hasattr(_atom_config, "set_current_atom_config"):
        _atom_config.set_current_atom_config = MagicMock()

# Sampler stub needs to be callable and return a usable object
_sampler_mod = sys.modules.get("atom.model_ops.sampler")
if _sampler_mod is not None:
    if not hasattr(_sampler_mod, "SAMPLER_EPS"):
        _sampler_mod.SAMPLER_EPS = 1e-6

    class _StubSampler:
        pass

    _sampler_mod.Sampler = _StubSampler

# ---------------------------------------------------------------------------
# Now safe to import model_runner and test dependencies
# ---------------------------------------------------------------------------

import math

import torch

from atom.kv_transfer.disaggregation.sim.sizing import sim_block_bytes
from atom.model_engine.model_runner import ModelRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_config(
    *,
    num_layers: int = 2,
    num_heads: int = 8,
    num_kv_heads: int = 8,
    hidden_size: int = 512,
    head_dim: int = 64,
    vocab_size: int = 1000,
    model_type: str = "llama",
) -> _types.SimpleNamespace:
    return _types.SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        vocab_size=vocab_size,
        model_type=model_type,
    )


def _sim_config(
    *,
    sim_isl: int = 2048,
    sim_concurrency: int = 8,
    block_size: int = 16,
    kv_cache_dtype: str = "bf16",
    tp_size: int = 1,
    hf_config=None,
) -> _types.SimpleNamespace:
    """Build a minimal Config-like namespace for the GPU-free sim paths."""
    hf = hf_config or _hf_config()
    return _types.SimpleNamespace(
        simulator=True,
        hf_config=hf,
        kv_cache_block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tp_size,
        sim_isl=sim_isl,
        sim_concurrency=sim_concurrency,
        max_model_len=sim_isl,
        max_num_seqs=sim_concurrency,
        max_num_batched_tokens=sim_isl,
        torch_dtype=torch.bfloat16,
        enforce_eager=True,
        mark_trace=False,
        speculative_config=None,
        torch_profiler_dir=None,
        parallel_config=_types.SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            data_parallel_size=1,
        ),
        # Fields that _get_num_blocks_sim may set on config.
        num_swa_blocks=0,
        swa_window_size=0,
        per_req_cache_equiv_blocks=0,
        num_per_req_cache_groups=0,
        num_kvcache_blocks=0,
    )


def _make_model_runner(config):
    """Construct a ModelRunner with only the sim-path attributes populated.

    Uses ``object.__new__`` to skip ``__init__`` (which needs GPU), then
    sets only the fields that the sim-path methods read.
    """
    runner = object.__new__(ModelRunner)
    runner.config = config
    runner._sim_skip_compute = True
    runner.block_size = config.kv_cache_block_size
    runner.kv_cache_dtype = config.kv_cache_dtype
    runner.world_size = config.tensor_parallel_size
    runner.device = torch.device("cpu")
    return runner


# ---------------------------------------------------------------------------
# _compute_block_bytes
# ---------------------------------------------------------------------------


class TestComputeBlockBytes:
    def test_returns_positive_int(self):
        config = _sim_config()
        runner = _make_model_runner(config)
        result = runner._compute_block_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_matches_sim_block_bytes(self):
        """Result matches the standalone sim_block_bytes helper."""
        config = _sim_config()
        runner = _make_model_runner(config)
        expected = sim_block_bytes(
            config.hf_config,
            config.kv_cache_block_size,
            kv_dtype=config.kv_cache_dtype,
            tp_size=config.tensor_parallel_size,
        )
        assert runner._compute_block_bytes() == expected

    def test_tp_sharding(self):
        """TP=2 halves the block bytes (8 KV heads sharded across 2 ranks)."""
        config_tp1 = _sim_config(tp_size=1)
        config_tp2 = _sim_config(tp_size=2)
        runner_tp1 = _make_model_runner(config_tp1)
        runner_tp2 = _make_model_runner(config_tp2)
        assert (
            runner_tp2._compute_block_bytes() == runner_tp1._compute_block_bytes() // 2
        )


# ---------------------------------------------------------------------------
# _get_num_blocks_sim
# ---------------------------------------------------------------------------


class TestGetNumBlocksSim:
    def test_returns_positive_num_kvcache_blocks(self):
        config = _sim_config(sim_isl=1024, sim_concurrency=4)
        runner = _make_model_runner(config)
        result = runner._get_num_blocks_sim()
        assert isinstance(result, dict)
        assert result["num_kvcache_blocks"] > 0

    def test_block_count_formula(self):
        """num_kvcache_blocks = ceil(sim_isl / block_size) * concurrency * 2."""
        config = _sim_config(sim_isl=2048, sim_concurrency=8, block_size=16)
        runner = _make_model_runner(config)
        result = runner._get_num_blocks_sim()
        blocks_per_req = math.ceil(2048 / 16)
        expected = blocks_per_req * 8 * 2
        assert result["num_kvcache_blocks"] == expected

    def test_swa_fields_zero(self):
        config = _sim_config()
        runner = _make_model_runner(config)
        result = runner._get_num_blocks_sim()
        assert result["num_swa_blocks"] == 0
        assert result["swa_window_size"] == 0
        assert result["per_req_cache_equiv_blocks"] == 0


# ---------------------------------------------------------------------------
# _run_model_sim
# ---------------------------------------------------------------------------


class TestRunModelSim:
    def test_logits_shape(self):
        """Returns logits with shape [num_tokens, vocab_size]."""
        config = _sim_config()
        runner = _make_model_runner(config)
        input_ids = torch.zeros(32, dtype=torch.long)
        logits, _ = runner._run_model_sim(input_ids)
        assert logits.shape == (32, config.hf_config.vocab_size)
        assert logits.dtype == torch.float32

    def test_hidden_states_shape(self):
        """Returns hidden_states with shape [num_tokens, hidden_size]."""
        config = _sim_config()
        runner = _make_model_runner(config)
        input_ids = torch.zeros(16, dtype=torch.long)
        _, hidden = runner._run_model_sim(input_ids)
        assert hidden.shape == (16, config.hf_config.hidden_size)
        assert hidden.dtype == config.torch_dtype

    def test_no_model_attribute_needed(self):
        """_run_model_sim works without self.model being set."""
        config = _sim_config()
        runner = _make_model_runner(config)
        assert not hasattr(runner, "model")
        input_ids = torch.zeros(4, dtype=torch.long)
        logits, hidden = runner._run_model_sim(input_ids)
        assert logits.shape[0] == 4
        assert hidden.shape[0] == 4

    def test_single_token(self):
        """Works with a single-token input."""
        config = _sim_config()
        runner = _make_model_runner(config)
        input_ids = torch.zeros(1, dtype=torch.long)
        logits, hidden = runner._run_model_sim(input_ids)
        assert logits.shape == (1, config.hf_config.vocab_size)
        assert hidden.shape == (1, config.hf_config.hidden_size)


# ---------------------------------------------------------------------------
# _allocate_kv_cache_sim
# ---------------------------------------------------------------------------


class TestAllocateKvCacheSim:
    def test_returns_true(self):
        config = _sim_config()
        runner = _make_model_runner(config)
        assert runner._allocate_kv_cache_sim(256) is True

    def test_sets_num_physical_blocks(self):
        config = _sim_config()
        runner = _make_model_runner(config)
        runner._allocate_kv_cache_sim(128)
        assert runner.num_physical_kvcache_blocks == 128
        assert config.num_kvcache_blocks == 128
