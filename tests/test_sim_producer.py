# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``sim`` KV connector — Phase 2: producer role.

GPU-free: the connector never touches CUDA/AITER.  Mirrors the style of
``test_sim_decode_connector.py`` (MagicMock config, real Sequence).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from atom.kv_transfer.disaggregation.sim.sim_connector import (
    SimConnector,
    SimConnectorScheduler,
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


def _producer_sched_config(**kv_extra) -> MagicMock:
    """Scheduler-side config for the producer role."""
    cfg = MagicMock()
    cfg.kv_transfer_config = {"kv_role": "kv_producer", **kv_extra}
    return cfg


def _producer_worker_config(**kv_extra) -> types.SimpleNamespace:
    """Worker-side config for the producer role."""
    return types.SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_producer", **kv_extra},
        hf_config=_hf_config(),
        kv_cache_block_size=16,
        kv_cache_dtype="bf16",
    )


def _make_seq(
    *,
    token_ids: list[int] | None = None,
    kv_transfer_params: dict | None = None,
    block_table: list[int] | None = None,
    output_tokens: list[int] | None = None,
) -> Sequence:
    if token_ids is None:
        token_ids = list(range(10))
    seq = Sequence(token_ids, block_size=16, kv_transfer_params=kv_transfer_params)
    if block_table is not None:
        seq.block_table = block_table
    if output_tokens is not None:
        seq.output_tokens = output_tokens
    return seq


@pytest.fixture()
def producer_sched() -> SimConnectorScheduler:
    return SimConnectorScheduler(_producer_sched_config())


# ---------------------------------------------------------------------------
# Scheduler: request_finished populates the handshake payload
# ---------------------------------------------------------------------------


class TestProducerRequestFinished:
    def test_handshake_payload_fields(self, producer_sched):
        """request_finished populates all required handshake fields."""
        seq = _make_seq(
            token_ids=[0] * 64,
            block_table=[10, 11, 12, 13],
            output_tokens=[42],
        )
        producer_sched.request_finished(seq)

        out = seq.kv_transfer_params_output
        assert out is not None
        assert out["remote_block_ids"] == [10, 11, 12, 13]
        assert out["remote_engine_id"] == producer_sched.engine_id
        assert out["remote_host"] == producer_sched.host_ip
        assert out["remote_handshake_port"] == producer_sched.handshake_port
        assert out["transfer_id"] == seq.id
        assert out["first_token_id"] == 42
        assert out["do_remote_prefill"] is True
        assert out["do_remote_decode"] is False
        assert out["tp_size"] == producer_sched.tp_size
        assert out["dp_rank"] == producer_sched.dp_rank

    def test_empty_output_tokens(self, producer_sched):
        """first_token_id is None when output_tokens is empty."""
        seq = _make_seq(block_table=[0], output_tokens=[])
        producer_sched.request_finished(seq)
        assert seq.kv_transfer_params_output["first_token_id"] is None

    def test_draft_token_ids(self, producer_sched):
        """spec_token_ids are forwarded as draft_token_ids."""
        seq = _make_seq(block_table=[0], output_tokens=[7])
        seq.spec_token_ids = [100, 101, 102]
        producer_sched.request_finished(seq)
        assert seq.kv_transfer_params_output["draft_token_ids"] == [100, 101, 102]

    def test_no_spec_tokens(self, producer_sched):
        """draft_token_ids is empty when no spec tokens exist."""
        seq = _make_seq(block_table=[0], output_tokens=[7])
        producer_sched.request_finished(seq)
        assert seq.kv_transfer_params_output["draft_token_ids"] == []

    def test_consumer_is_noop(self):
        """request_finished is a no-op on the consumer side."""
        cfg = MagicMock()
        cfg.kv_transfer_config = {"kv_role": "kv_consumer"}
        sched = SimConnectorScheduler(cfg)
        seq = _make_seq(block_table=[1])
        sched.request_finished(seq)
        assert seq.kv_transfer_params_output is None

    def test_queues_send_for_worker(self, producer_sched):
        """request_finished queues the seq.id for the worker's done_sending."""
        seq = _make_seq(block_table=[5, 6], output_tokens=[1])
        producer_sched.request_finished(seq)
        assert seq.id in producer_sched._reqs_need_send


# ---------------------------------------------------------------------------
# Worker: producer get_finished returns done_sending once then empty
# ---------------------------------------------------------------------------


class TestProducerWorkerGetFinished:
    def test_done_sending_once_then_empty(self):
        """Producer sends are instant; get_finished yields them once."""
        worker = SimConnector(_producer_worker_config())
        meta = ConnectorMetadata()
        meta.reqs_to_send = {42: 0.0, 99: 0.0}
        worker.start_load_kv(meta)

        done_sending, done_recving = worker.get_finished()
        assert done_sending == {42, 99}
        assert done_recving == set()

        # Second call: buffer cleared.
        done_sending2, done_recving2 = worker.get_finished()
        assert done_sending2 == set()
        assert done_recving2 == set()

    def test_no_sends_no_output(self):
        """Worker with no pending sends returns empty sets."""
        worker = SimConnector(_producer_worker_config())
        assert worker.get_finished() == (set(), set())


# ---------------------------------------------------------------------------
# Scheduler: build_connector_meta drains send queue
# ---------------------------------------------------------------------------


class TestProducerBuildConnectorMeta:
    def test_sends_drained(self, producer_sched):
        """build_connector_meta carries finished sends and drains the queue."""
        seq = _make_seq(block_table=[0], output_tokens=[1])
        producer_sched.request_finished(seq)

        meta = producer_sched.build_connector_meta()
        assert seq.id in meta.reqs_to_send
        # Queue is drained after build.
        assert producer_sched._reqs_need_send == {}
