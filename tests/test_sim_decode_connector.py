# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``sim`` KV connector (Phase 1: sim-decode consumer).

GPU-free: the connector never touches CUDA/AITER, and topology resolution
falls back to rank 0 / size 1 when the distributed groups are unavailable.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from atom.kv_transfer.disaggregation.sim.sim_connector import (
    SimConnector,
    SimConnectorScheduler,
)
from atom.kv_transfer.disaggregation.sim.sizing import (
    kv_dtype_itemsize,
    sim_block_bytes,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.model_engine.sequence import Sequence

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _hf_config() -> types.SimpleNamespace:
    """A tiny weight-free HF-like config for byte sizing."""
    return types.SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_size=512,
        head_dim=64,
    )


def _sched_config(*, kv_role: str = "kv_consumer", **kv_extra) -> MagicMock:
    """Scheduler-side config (MagicMock is fine — no sizing on this path)."""
    cfg = MagicMock()
    cfg.kv_transfer_config = {"kv_role": kv_role, **kv_extra}
    return cfg


def _worker_config(
    *, kv_role: str = "kv_consumer", **kv_extra
) -> types.SimpleNamespace:
    """Worker-side config with a real numeric hf_config for byte sizing."""
    return types.SimpleNamespace(
        kv_transfer_config={"kv_role": kv_role, **kv_extra},
        hf_config=_hf_config(),
        kv_cache_block_size=16,
        kv_cache_dtype="bf16",
    )


def _make_seq(
    *,
    token_ids: list[int] | None = None,
    kv_transfer_params: dict | None = None,
    block_table: list[int] | None = None,
) -> Sequence:
    if token_ids is None:
        token_ids = list(range(10))
    seq = Sequence(token_ids, block_size=16, kv_transfer_params=kv_transfer_params)
    if block_table is not None:
        seq.block_table = block_table
    return seq


@pytest.fixture()
def consumer_sched() -> SimConnectorScheduler:
    return SimConnectorScheduler(_sched_config(kv_role="kv_consumer"))


# ---------------------------------------------------------------------------
# Scheduler: get_num_new_matched_tokens forces the skip-prefill path
# ---------------------------------------------------------------------------


class TestGetNumNewMatchedTokens:
    def test_plain_request_forced_remote(self, consumer_sched):
        """A plain request (no do_remote_prefill) still skips prefill."""
        seq = _make_seq(token_ids=[0] * 128, kv_transfer_params=None)
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 128
        assert needs_load is True

    def test_idempotent_second_call(self, consumer_sched):
        seq = _make_seq(token_ids=[0] * 64)
        consumer_sched.get_num_new_matched_tokens(seq)
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 0
        assert needs_load is False

    def test_autoprefill_disabled(self):
        sched = SimConnectorScheduler(
            _sched_config(kv_role="kv_consumer", sim_autoprefill=False)
        )
        seq = _make_seq(token_ids=[0] * 32)
        assert sched.get_num_new_matched_tokens(seq) == (0, False)

    def test_producer_role_no_match(self):
        sched = SimConnectorScheduler(_sched_config(kv_role="kv_producer"))
        seq = _make_seq(token_ids=[0] * 32)
        assert sched.get_num_new_matched_tokens(seq) == (0, False)


# ---------------------------------------------------------------------------
# Scheduler: update_state_after_alloc injects T0 + queues the recv
# ---------------------------------------------------------------------------


class TestUpdateStateAfterAlloc:
    def test_sets_first_token_and_queues(self, consumer_sched):
        seq = _make_seq(token_ids=[0] * 20, block_table=[10, 11, 12])
        consumer_sched.get_num_new_matched_tokens(seq)  # tags the seq
        consumer_sched.update_state_after_alloc(seq)

        assert seq.kv_transfer_params["first_token_id"] == 0
        assert seq.id in consumer_sched._reqs_need_recv
        assert consumer_sched._reqs_need_recv[seq.id][1] == [10, 11, 12]
        # bidirectional transfer_id mapping is consistent
        tid = consumer_sched.request_id_to_transfer_id[seq.id]
        assert consumer_sched.transfer_id_to_request_id[tid] == seq.id

    def test_untagged_seq_not_queued(self, consumer_sched):
        """Without the get_num_new_matched_tokens tag, nothing is queued."""
        seq = _make_seq(block_table=[1, 2])
        consumer_sched.update_state_after_alloc(seq)
        assert seq.id not in consumer_sched._reqs_need_recv

    def test_custom_first_token_id(self):
        sched = SimConnectorScheduler(
            _sched_config(kv_role="kv_consumer", sim_first_token_id=5)
        )
        seq = _make_seq(token_ids=[0] * 8, block_table=[0])
        sched.get_num_new_matched_tokens(seq)
        sched.update_state_after_alloc(seq)
        assert seq.kv_transfer_params["first_token_id"] == 5


# ---------------------------------------------------------------------------
# Scheduler: build_connector_meta drains the queue into ConnectorMetadata
# ---------------------------------------------------------------------------


class TestBuildConnectorMeta:
    def test_empty(self, consumer_sched):
        meta = consumer_sched.build_connector_meta()
        assert isinstance(meta, ConnectorMetadata)
        assert meta.reqs_to_recv == {}

    def test_drains_and_carries_block_ids(self, consumer_sched):
        seq = _make_seq(token_ids=[0] * 20, block_table=[4, 5, 6])
        consumer_sched.get_num_new_matched_tokens(seq)
        consumer_sched.update_state_after_alloc(seq)

        meta = consumer_sched.build_connector_meta()
        assert seq.id in meta.reqs_to_recv
        assert meta.reqs_to_recv[seq.id].local_block_ids == [4, 5, 6]
        assert consumer_sched._reqs_need_recv == {}

    def test_request_finished_is_noop(self, consumer_sched):
        seq = _make_seq(block_table=[1])
        assert consumer_sched.request_finished(seq) is None
        assert seq.kv_transfer_params_output is None


# ---------------------------------------------------------------------------
# Worker: instant completion (no transfer rate)
# ---------------------------------------------------------------------------


def _meta_with_recv(req_id: int, block_ids: list[int]) -> ConnectorMetadata:
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id=req_id,
        local_block_ids=block_ids,
        kv_transfer_params={"remote_block_ids": block_ids, "transfer_id": 0},
    )
    return meta


class TestWorkerInstant:
    def test_recv_completes_once_then_empty(self):
        worker = SimConnector(_worker_config(kv_role="kv_consumer"))
        worker.start_load_kv(_meta_with_recv(req_id=7, block_ids=[0, 1, 2]))

        done_sending, done_recving = worker.get_finished()
        assert done_sending == set()
        assert done_recving == {7}

        # Second call: nothing pending -> both empty.
        assert worker.get_finished() == (set(), set())

    def test_never_reports_sending(self):
        worker = SimConnector(_worker_config(kv_role="kv_consumer"))
        worker.start_load_kv(_meta_with_recv(req_id=1, block_ids=[0]))
        done_sending, _ = worker.get_finished()
        assert done_sending == set()

    def test_register_kv_caches_noop(self):
        worker = SimConnector(_worker_config(kv_role="kv_consumer"))
        # Accepts the transfer_tensors + num_blocks positional/kw args.
        assert worker.register_kv_caches({}, None, num_blocks=8) is None
        assert worker.get_finished_recv_blocks() == []


# ---------------------------------------------------------------------------
# Worker: rate-paced completion
# ---------------------------------------------------------------------------


class TestWorkerTransferRate:
    def test_fixed_delay_defers_completion(self, monkeypatch):
        worker = SimConnector(
            _worker_config(kv_role="kv_consumer", sim_transfer_delay_ms=500)
        )

        clock = {"t": 1000.0}
        monkeypatch.setattr(
            "atom.kv_transfer.disaggregation.sim.sim_connector.time.monotonic",
            lambda: clock["t"],
        )
        worker.start_load_kv(_meta_with_recv(req_id=3, block_ids=[0, 1]))

        # Not yet ready (0.5s delay not elapsed).
        assert worker.get_finished() == (set(), set())

        # Advance past the deadline.
        clock["t"] += 0.6
        assert worker.get_finished() == (set(), {3})

    def test_bytes_per_sec_derives_delay(self, monkeypatch):
        worker = SimConnector(
            _worker_config(kv_role="kv_consumer", sim_transfer_gbps=1.0)
        )
        assert worker._block_bytes > 0

        clock = {"t": 0.0}
        monkeypatch.setattr(
            "atom.kv_transfer.disaggregation.sim.sim_connector.time.monotonic",
            lambda: clock["t"],
        )
        worker.start_load_kv(_meta_with_recv(req_id=9, block_ids=[0, 1, 2, 3]))

        expected = (4 * worker._block_bytes) / 1e9
        assert worker._pending_recv[9] == pytest.approx(expected)

        # Just before vs after the derived deadline.
        clock["t"] = expected - 1e-9
        assert worker.get_finished() == (set(), set())
        clock["t"] = expected
        assert worker.get_finished() == (set(), {9})


# ---------------------------------------------------------------------------
# sizing helpers
# ---------------------------------------------------------------------------


class TestSizing:
    def test_block_bytes_matches_formula(self):
        hf = _hf_config()
        # 2(K+V) * layers(2) * block(16) * kv_heads(8) * head_dim(64) * 2(bf16)
        expected = 2 * 2 * 16 * 8 * 64 * 2
        assert sim_block_bytes(hf, 16, "bf16", tp_size=1) == expected

    def test_tp_shards_kv_heads(self):
        hf = _hf_config()
        full = sim_block_bytes(hf, 16, "bf16", tp_size=1)
        sharded = sim_block_bytes(hf, 16, "bf16", tp_size=2)
        assert sharded == full // 2

    def test_dtype_itemsize(self):
        assert kv_dtype_itemsize("fp8") == 1
        assert kv_dtype_itemsize("bf16") == 2
        assert kv_dtype_itemsize("float32") == 4
        assert kv_dtype_itemsize("torch.bfloat16") == 2
        assert kv_dtype_itemsize(None) == 2
        assert kv_dtype_itemsize("auto") == 2

    def test_block_bytes_always_positive(self):
        assert sim_block_bytes(types.SimpleNamespace(), 16, "bf16") >= 1
