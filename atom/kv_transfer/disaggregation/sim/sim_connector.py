# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
The ``sim`` KV connector — a compute-free stand-in for a real P/D transfer.

**Consumer / decode role** (Phase 1):

- ``SimConnectorScheduler`` unconditionally reports every incoming request as
  "remotely matched" (``sim_autoprefill``), so the scheduler parks it in
  ``WAITING_FOR_REMOTE_KVS``, allocates the full ISL of KV blocks, and skips
  the prefill forward pass.
- ``SimConnector`` (worker) fabricates instant — or rate-paced — receive
  completion, so the scheduler flips the request to ``RUNNING`` and injects a
  synthetic first token.  Decode then runs as **real GPU compute** over the
  (garbage) KV blocks.  Decode perf is content-independent, so metrics are
  faithful.

**Producer / prefill role** (Phase 2):

- ``SimConnectorScheduler.request_finished`` populates the P/D handshake
  payload (``seq.kv_transfer_params_output``) so the proxy can relay block
  metadata to the decode instance — mirroring the MoRIIO producer path but
  without any RDMA.
- ``SimConnector.get_finished`` reports producer sends as instantly done so
  the scheduler can free deferred blocks.

ISL and concurrency are properties of the incoming request stream (set by the
benchmark client: ``--random-input-len`` and ``--max-concurrency``).  The
optional **transfer-rate** knob paces per-request completion to emulate
prefill->decode transfer latency (see ``sim_transfer_gbps`` /
``sim_transfer_delay_ms``).
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

        # Producer side: send req_ids reported as finished (instant, no RDMA).
        self._done_sending: set[Any] = set()

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
        return

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Record pending receives or instant sends depending on role.

        Consumer: stamps each receive with a ``ready_at`` deadline.
        Producer: marks sends as instantly complete (no RDMA).
        """
        if metadata is None:
            return
        if self.is_producer:
            # Producer sends are instant — record transfer IDs as done.
            for req_id in metadata.reqs_to_send:
                self._done_sending.add(req_id)
            if metadata.reqs_to_send:
                logger.debug(
                    "SimConnector queued %d instant send(s): %s",
                    len(metadata.reqs_to_send),
                    list(metadata.reqs_to_send.keys()),
                )
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

        Producer: returns accumulated done sends and clears the buffer.
        Consumer: ``done_sending`` is always empty.  A receive is reported
        done once its ``ready_at`` deadline has passed (immediate when no
        transfer rate is configured).
        """
        if self.is_producer:
            done_sending = self._done_sending.copy()
            self._done_sending.clear()
            return done_sending, set()
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

        # Producer: finished sends queued for the worker via build_connector_meta.
        self._reqs_need_send: dict[ReqId, float] = {}

        # Topology metadata for the producer handshake payload.
        # Resolved GPU-free: missing parallel_config / get_ip fall back to
        # safe defaults so unit tests and GPU-free hosts still construct.
        self.host_ip = self._resolve_host_ip()
        self.tp_size = getattr(
            getattr(config, "parallel_config", None), "tensor_parallel_size", 1
        ) or getattr(config, "tensor_parallel_size", 1)
        self.dp_rank = getattr(
            getattr(config, "parallel_config", None), "data_parallel_rank", 0
        )
        self.handshake_port: int = kv_cfg.get("handshake_port", 0)
        self.engine_id = f"{self.host_ip}:{self.handshake_port}"

        logger.info(
            "SimConnectorScheduler initialised: is_producer=%s sim_autoprefill=%s "
            "first_token_id=%d engine_id=%s tp_size=%d dp_rank=%d",
            self.is_producer,
            self.sim_autoprefill,
            self.first_token_id,
            self.engine_id,
            self.tp_size,
            self.dp_rank,
        )

    @staticmethod
    def _resolve_host_ip() -> str:
        """Return the host IP, falling back to ``127.0.0.1`` on GPU-free hosts."""
        try:
            from atom.utils.network import get_ip

            return str(get_ip())
        except Exception:  # noqa: BLE001
            return "127.0.0.1"

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
        """Snapshot pending receives/sends for the worker; clear the queues."""
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

        # Producer: pass finished sends to the worker for instant completion.
        if self._reqs_need_send:
            meta.reqs_to_send = dict(self._reqs_need_send)
            logger.debug(
                "SimConnectorScheduler built meta with %d send(s)",
                len(self._reqs_need_send),
            )
            self._reqs_need_send.clear()

        return meta

    def request_finished(self, seq: Sequence) -> None:
        """Emit the P/D handshake payload on the producer side.

        On the producer side, populates ``seq.kv_transfer_params_output`` with
        the same fields the MoRIIO connector emits (moriio_connector.py lines
        978-996) so the proxy can relay block metadata to the decode instance.
        Also queues the transfer_id for the worker to report as done_sending.

        On the consumer side this is a no-op.
        """
        if not self.is_producer:
            return

        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        drafts = getattr(seq, "spec_token_ids", None)
        draft_token_ids = (
            [int(x) for x in drafts] if drafts is not None and len(drafts) else []
        )
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": list(seq.block_table),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "remote_handshake_port": self.handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "draft_token_ids": draft_token_ids,
        }

        # Queue for the worker to report as done_sending on the next step.
        self._reqs_need_send[seq.id] = time.monotonic()
        logger.debug(
            "SimConnectorScheduler producer request_finished: seq=%s "
            "blocks=%d first_token=%s",
            seq.id,
            len(seq.block_table),
            first_token_id,
        )
