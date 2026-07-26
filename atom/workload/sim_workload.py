# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Synthetic workload generator for the prefill/decode simulator.

Injects ``--sim-concurrency`` requests of ``[0] * sim_isl`` tokens into the
engine via its ``io_processor`` / ``core_mgr`` pathway, paced by
``--sim-rate`` (requests/sec).  Each request carries
``kv_transfer_params={"do_remote_prefill": False}`` so the producer
connector emits the handshake payload without expecting a remote decode
instance.

Started from the api_server hook (``start_sim_workload``) after the engine
is fully constructed.
"""

from __future__ import annotations

import logging
import threading
import time

from atom.sampling_params import SamplingParams

logger = logging.getLogger("atom")


def start_sim_workload(engine, config, engine_args) -> threading.Thread:
    """Launch a background thread that injects simulated requests.

    Args:
        engine: A fully-constructed ``LLMEngine`` instance.
        config: The engine's ``Config`` (``engine.config``).
        engine_args: The ``EngineArgs`` dataclass that holds
            ``sim_isl``, ``sim_concurrency``, and ``sim_rate``.

    Returns:
        The daemon ``Thread`` (already started).
    """
    sim_isl = getattr(engine_args, "sim_isl", 2048)
    sim_concurrency = getattr(engine_args, "sim_concurrency", 8)
    sim_rate = getattr(engine_args, "sim_rate", None)

    thread = threading.Thread(
        target=_inject_requests,
        args=(engine, sim_isl, sim_concurrency, sim_rate),
        name="sim-workload",
        daemon=True,
    )
    thread.start()
    logger.info(
        "sim_workload: started background thread "
        "(isl=%d, concurrency=%d, rate=%s req/s)",
        sim_isl,
        sim_concurrency,
        sim_rate if sim_rate is not None else "closed-loop",
    )
    return thread


def _inject_requests(
    engine,
    sim_isl: int,
    sim_concurrency: int,
    sim_rate: float | None,
) -> None:
    """Thread target: inject *sim_concurrency* dummy requests into *engine*."""
    delay = 1.0 / sim_rate if sim_rate and sim_rate > 0 else 0.0
    tokens = [0] * sim_isl
    sampling_params = SamplingParams(max_tokens=1, ignore_eos=True)
    kv_transfer_params = {"do_remote_prefill": False}

    for i in range(sim_concurrency):
        req = engine.io_processor.preprocess(
            tokens,
            sampling_params,
            kv_transfer_params=kv_transfer_params,
            request_id=f"sim-{i}",
        )
        engine.core_mgr.add_request([req])
        logger.debug("sim_workload: injected request sim-%d (%d tokens)", i, sim_isl)
        if delay > 0 and i < sim_concurrency - 1:
            time.sleep(delay)

    logger.info("sim_workload: finished injecting %d requests", sim_concurrency)
