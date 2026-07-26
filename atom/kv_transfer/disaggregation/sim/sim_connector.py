# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
The ``sim`` KV connector — a compute-free stand-in for a real P/D transfer.

Phase 1 (this module) implements the **consumer / decode** role.  It lets a
real decode server measure decode performance *without* a prefill server, a
router, or any RDMA:

- ``SimConnectorScheduler`` unconditionally reports every incoming request as
  "remotely matched" (``sim_autoprefill``), so the scheduler parks it in
  ``WAITING_FOR_REMOTE_KVS``, allocates the full ISL of KV blocks, and skips
  the prefill forward pass.
- ``SimConnector`` (worker) fabricates instant — or rate-paced — receive
  completion, so the scheduler flips the request to ``RUNNING`` and injects a
  synthetic first token.  Decode then runs as **real GPU compute** over the
  (garbage) KV blocks.  Decode perf is content-independent, so metrics are
  faithful.

ISL and concurrency are properties of the incoming request stream (set by the
benchmark client: ``--random-input-len`` and ``--max-concurrency``).  The
optional **transfer-rate** knob paces per-request completion to emulate
prefill->decode transfer latency (see ``sim_transfer_gbps`` /
``sim_transfer_delay_ms``).

The producer role is intentionally left as a Phase 2 extension.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.sim.sizing import sim_block_bytes
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    ReqId,
    TransferId,
)
from atom.model_engine.sequence import Sequence

logger = logging.getLogger("atom")

# Config keys read from ``config.kv_transfer_config``.
_KEY_ROLE = "kv_role"
_KEY_AUTOPREFILL = "sim_autoprefill"
_KEY_FIRST_TOKEN = "sim_first_token_id"
_KEY_TRANSFER_GBPS = "sim_transfer_gbps"
_KEY_TRANSFER_DELAY_MS = "sim_transfer_delay_ms"


# ===================================================================
# SimConnector — worker-side (one instance per TP rank)
# ===================================================================


class SimConnector(KVConnectorBase):
    """Worker-side ``sim`` connector: fabricates KV receive completion.

    No RDMA, no handshake, no proxy.  For each pending receive it computes a
    ``ready_at`` deadline (immediate unless a transfer rate is configured) and
    reports the request as done once that deadline passes.
    """

    def __init__(self, config: Config) -> None:
        kv_cfg = getattr(config, "kv_transfer_config", None) or {}
        self.is_producer = kv_cfg.get(_KEY_ROLE, "kv_producer") == "kv_producer"

        # Rank/topology defaults; guarded so a GPU-free host still constructs.
        self.tp_rank, self.tp_size, self.dp_rank = self._resolve_topology()

        # --- Transfer-rate knob -------------------------------------------
        # A fixed per-request delay takes precedence; otherwise a bytes/s
        # budget derives the delay from each request's KV footprint; otherwise
        # completion is instant (current default behaviour).
        delay_ms = kv_cfg.get(_KEY_TRANSFER_DELAY_MS)
        gbps = kv_cfg.get(_KEY_TRANSFER_GBPS)
        self._fixed_delay_s: float = float(delay_ms) / 1000.0 if delay_ms else 0.0
        self._rate_bytes_per_s: float = float(gbps) * 1e9 if gbps else 0.0

        # Byte size of one locally-allocated KV block (per rank), for the
        # bytes/s pacing path.  Derived weight-free from the HF config.
        self._block_bytes: int = sim_block_bytes(
            getattr(config, "hf_config", None),
            getattr(config, "kv_cache_block_size", 16),
            getattr(config, "kv_cache_dtype", None),
            tp_size=self.tp_size,
        )

        # req_id -> monotonic deadline at which the recv is considered done.
        self._pending_recv: dict[Any, float] = {}

        logger.info(
            "SimConnector[worker] initialised: is_producer=%s tp_rank=%d/%d "
            "fixed_delay=%.3fs rate=%.1fGB/s block_bytes=%d",
            self.is_producer,
            self.tp_rank,
            self.tp_size,
            self._fixed_delay_s,
            self._rate_bytes_per_s / 1e9,
            self._block_bytes,
        )

    @staticmethod
    def _resolve_topology() -> tuple[int, int, int]:
        """Return ``(tp_rank, tp_size, dp_rank)``, defaulting to rank 0 / size 1.

        Guarded with try/except so the connector constructs on a GPU-free host
        where the distributed groups are not initialised.
        """
        try:
            from aiter.dist.parallel_state import get_dp_group, get_tp_group

            # int() coercion also collapses stubbed/MagicMock groups to the
            # defaults (non-int raises TypeError -> caught below).
            return (
                int(get_tp_group().rank_in_group),
                int(get_tp_group().world_size),
                int(get_dp_group().rank_in_group),
            )
        except Exception:  # noqa: BLE001 — GPU-free / uninitialised groups
            return 0, 1, 0

    def _delay_for(self, num_blocks: int) -> float:
        """Per-request completion delay (seconds) for *num_blocks* blocks."""
        if self._fixed_delay_s > 0.0:
            return self._fixed_delay_s
        if self._rate_bytes_per_s > 0.0:
            transferred = max(0, int(num_blocks)) * self._block_bytes
            return transferred / self._rate_bytes_per_s
        return 0.0

    # --- KVConnectorBase API ---------------------------------------------

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        """No-op: the sim connector never touches real KV memory."""
        return None

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Record pending receives, stamping each with a ``ready_at`` deadline."""
        if self.is_producer or metadata is None:
            return
        now = time.monotonic()
        for req_id, meta in metadata.reqs_to_recv.items():
            num_blocks = len(getattr(meta, "local_block_ids", None) or [])
            self._pending_recv[req_id] = now + self._delay_for(num_blocks)
        if metadata.reqs_to_recv:
            logger.debug(
                "SimConnector queued %d recv(s): %s",
                len(metadata.reqs_to_recv),
                list(metadata.reqs_to_recv.keys()),
            )

    def get_finished(self) -> tuple[set, set]:
        """Return ``(done_sending, done_recving)``.

        Consumer never sends, so ``done_sending`` is always empty.  A receive
        is reported done once its ``ready_at`` deadline has passed (immediate
        when no transfer rate is configured).
        """
        if self.is_producer:
            return set(), set()
        now = time.monotonic()
        done_recving = {
            req_id for req_id, ready_at in self._pending_recv.items() if ready_at <= now
        }
        for req_id in done_recving:
            del self._pending_recv[req_id]
        return set(), done_recving

    def get_finished_recv_blocks(self) -> list[int]:
        """No GPU memory fence needed — the KV was never really transferred."""
        return []


# ===================================================================
# SimConnectorScheduler — scheduler-side
# ===================================================================


class SimConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side ``sim`` connector: forces the skip-prefill path.

    In sim-decode mode (``sim_autoprefill``, default ``True`` for a consumer)
    every waiting request is reported as fully remotely-matched, which routes
    it into ``WAITING_FOR_REMOTE_KVS`` and skips its prefill forward.
    """

    def __init__(self, config: Config) -> None:
        kv_cfg = getattr(config, "kv_transfer_config", None) or {}
        self.is_producer = kv_cfg.get(_KEY_ROLE, "kv_producer") == "kv_producer"

        # Default: auto-prefill on for a consumer (there is no real producer to
        # supply ``do_remote_prefill``).  Explicit config overrides.
        self.sim_autoprefill: bool = bool(
            kv_cfg.get(_KEY_AUTOPREFILL, not self.is_producer)
        )
        self.first_token_id: int = int(kv_cfg.get(_KEY_FIRST_TOKEN, 0))

        # Pending receives queued for the worker: req_id -> (seq, block_ids).
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}
        self._next_transfer_id: int = 0

        logger.info(
            "SimConnectorScheduler initialised: is_producer=%s sim_autoprefill=%s "
            "first_token_id=%d",
            self.is_producer,
            self.sim_autoprefill,
            self.first_token_id,
        )

    # --- KVConnectorSchedulerBase API ------------------------------------

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        """Report the whole prompt as remotely matched in sim-decode mode.

        Returns ``(num_prompt_tokens, True)`` exactly once per sequence so the
        scheduler parks it for remote KV and skips the prefill forward.  The
        ``kv_async_tagged`` guard prevents re-triggering on later passes.
        """
        if (
            self.is_producer
            or not self.sim_autoprefill
            or hasattr(seq, "kv_async_tagged")
        ):
            return 0, False
        seq.kv_async_tagged = True
        return len(seq.prompt_token_ids), True

    def update_state_after_alloc(self, seq: Sequence) -> None:
        """After block allocation, queue the (fabricated) remote receive.

        Sets a synthetic ``first_token_id`` (injected by the scheduler when the
        transfer "completes") and dummy remote metadata pointing at the local
        blocks, then queues the request for the worker.
        """
        if self.is_producer or not getattr(seq, "kv_async_tagged", False):
            return

        block_ids = list(seq.block_table)
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1
        self.transfer_id_to_request_id[transfer_id] = seq.id
        self.request_id_to_transfer_id[seq.id] = transfer_id

        params = dict(seq.kv_transfer_params or {})
        params.setdefault("first_token_id", self.first_token_id)
        # Dummy remote metadata: the "remote" blocks are the local ones, and
        # there is no host/port to reach (the worker never uses them).
        params["remote_block_ids"] = block_ids
        params["transfer_id"] = transfer_id
        params["do_remote_prefill"] = False
        seq.kv_transfer_params = params

        self._reqs_need_recv[seq.id] = (seq, block_ids)
        logger.debug(
            "SimConnectorScheduler queued req %s for sim recv (%d blocks, tid=%d)",
            seq.id,
            len(block_ids),
            transfer_id,
        )

    def build_connector_meta(self) -> ConnectorMetadata:
        """Snapshot pending receives for the worker; clear the queue."""
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = dict(self.request_id_to_transfer_id)
        for req_id, (seq, block_ids) in self._reqs_need_recv.items():
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=seq.kv_transfer_params or {},
            )
        if self._reqs_need_recv:
            logger.debug(
                "SimConnectorScheduler built meta with %d recv(s)",
                len(self._reqs_need_recv),
            )
        self._reqs_need_recv.clear()
        return meta

    def request_finished(self, seq: Sequence) -> None:
        """Consumer side has nothing to emit when a request finishes."""
        return None
