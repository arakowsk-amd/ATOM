# Sim Producer & Full P/D Pipeline Guide

This guide covers the **producer** side of the `sim` KV connector (Phase 2) and how to
stand up the **full disaggregated prefill/decode (P/D) pipeline** end-to-end —
router → sim-producer → sim-consumer — **without any GPU prefill compute and without
RDMA**. The handshake that a real Mooncake/MoRIIO transfer would carry over RDMA is
instead relayed as plain HTTP JSON by the **atomesh** router.

For the **consumer / decode** side (skip-prefill mechanics, transfer-rate knobs, TP
sharding of KV accounting), see the companion doc
[`sim_decode_connector_guide.md`](./sim_decode_connector_guide.md). This guide does
**not** duplicate that content — it focuses on the producer and the wired-together
pipeline.

---

## 1. Capability map

What the sim P/D simulator can and cannot do today (verified on an MI355 node,
Llama-3.1-8B-Instruct-FP8-KV):

| Capability | Status | Notes |
|---|---|---|
| Decode-perf measurement **without a prefill server** (Phase 1 consumer) | **Works, faithful** | Benchmark drives ISL via `--random-input-len`, concurrency via `--max-concurrency`. Paged attention reads all ISL blocks regardless of KV contents, so TPOT/ITL/throughput are real. |
| Decode **TP** (`-tp N`) | **Works, real/faithful** | Real GPU decode; KV heads are sharded across the real TP ranks. |
| Full sim pipeline (router → sim-producer → sim-consumer decode) over HTTP | **Works** | No RDMA; the P/D handshake is HTTP JSON relayed by the atomesh router. |
| Sim producer, **GPU-compute-free** (Phase 2) | **Works** | ~0 VRAM; `load_dummy=empty`; `enforce_eager=True`; `world_size` forced to 1; device=cpu. |
| Transfer-rate knob (`sim_transfer_delay_ms`, `sim_transfer_gbps`) | **Modeled** | Paces per-request recv completion → shifts TTFT (and, closed-loop, throughput). |
| Prefill **TP/DP simulation** (`--sim-prefill-tp` / `--sim-prefill-dp`) | **NOT modeled** | Unused stubs (see §7). Advertised logical topology is **not physically realized**; in compute-skip mode `world_size` is forced to 1. |

The core idea: **decode performance is content-independent**, so you can measure a
real decode server's TPOT/ITL/throughput at any ISL and concurrency using garbage KV
blocks — no real prefill needed. The sim producer then lets you exercise the full P/D
control path (routing, handshake, block-metadata relay) with **zero** prefill GPU cost.

---

## 2. Launch the sim producer (`simulator_server`)

The producer is a GPU-compute-free stand-in for a prefill instance. Launch it with the
dedicated entrypoint, which auto-injects `--simulator`:

```bash
cd /app/ATOM
export AITER_LOG_LEVEL=WARNING       # required per CLAUDE.md — suppresses aiter log flood
export HIP_VISIBLE_DEVICES=1         # any free GPU index; it stays ~0 VRAM
python -m atom.entrypoints.simulator_server \
  --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
  --kv_cache_dtype fp8 \
  --sim-concurrency 0 \
  --host 0.0.0.0 --server-port 8010
```

What `--simulator` forces (see `arg_utils.py::_get_engine_kwargs`):

- `Config.simulator=True`
- `load_dummy="empty"` — no real weights are read.
- `enforce_eager=True` — no torch.compile / CUDAGraph.
- Injects `{"kv_connector":"sim","kv_role":"kv_producer"}` into `kv_transfer_config`
  **unless** you passed your own.

In compute-skip mode the model runner (`_setup_device_and_distributed_sim`) sets
`device=cpu` and `world_size=1`, does **no** NCCL init and **no** `set_device`, and KV
is metadata-only — so the process consumes ~0 VRAM even though a GPU is visible.
`--sim-concurrency 0` means the producer does **not** self-inject a synthetic load; it
simply waits for the router to drive requests through it.

The producer's `SimConnectorScheduler.request_finished` populates
`seq.kv_transfer_params_output` — the P/D handshake payload (`do_remote_prefill=True`,
`remote_block_ids`, `remote_engine_id`, `tp_size`, `dp_rank`, `first_token_id`, …) —
mirroring the MoRIIO producer path but with no RDMA. `get_finished` reports each send
as instantly `done_sending`.

---

## 3. Stand up the full pipeline

Three processes on one node. Physical GPU placement is via per-process
`HIP_VISIBLE_DEVICES`. In the reference setup: consumer on index 0 (real decode,
~240 GB KV), producer on index 1 (compute-free), router on CPU.

> Per CLAUDE.md, clear the compile cache before (re)launching a server after any code
> change: `rm -rf /root/.cache/atom/*`. Start servers backgrounded to a log, then
> **verify with `rocm-smi --showmemuse` (VRAM% > 0 on the consumer GPU), not just
> `/health`.**

### 3a. Sim-consumer (real decode)

```bash
cd /app/ATOM
export AITER_LOG_LEVEL=WARNING
export HIP_VISIBLE_DEVICES=0
python -m atom.entrypoints.openai_server \
  --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
  --load_dummy empty \
  --kv_cache_dtype fp8 \
  -tp 1 \
  --host 0.0.0.0 --server-port 8020 \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"sim"}' \
  > /tmp/consumer.log 2>&1 &
```

`--load_dummy empty` gives real decode compute with dummy weights (outputs are
garbage, metrics are faithful). Use your own real `-tp` here. On the reference node
this yields `num_kvcache_blocks=239148` (~240 GB KV at fp8, block_bytes=1081344).

### 3b. Sim-producer (compute-free) — see §2

```bash
cd /app/ATOM
export AITER_LOG_LEVEL=WARNING
export HIP_VISIBLE_DEVICES=1
python -m atom.entrypoints.simulator_server \
  --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
  --kv_cache_dtype fp8 \
  --sim-concurrency 0 \
  --host 0.0.0.0 --server-port 8010 \
  > /tmp/producer.log 2>&1 &
```

### 3c. Router (atomesh)

Set `NODE_IP` to this node's address (both endpoints are on the same node here). This
is the same launch shape as `recipes/pd_disaggregation_guide.md`, minus the RDMA
handshake_port (the sim connector needs none).

```bash
NODE_IP=10.235.26.26
atomesh launch \
  --host 0.0.0.0 --port 8000 \
  --pd-disaggregation \
  --prefill "http://${NODE_IP}:8010" \
  --decode  "http://${NODE_IP}:8020" \
  --policy random \
  --backend atom \
  --log-level info \
  --disable-health-check \
  --disable-circuit-breaker \
  > /tmp/router.log 2>&1 &
```

Once all three are up (`curl :8000/health`, `:8010/health`, `:8020/health` and
VRAM% > 0 on the consumer GPU), send all client traffic to the **router** at
`http://<NODE_IP>:8000`.

---

## 4. Benchmark against the router

Drive the router endpoint with the standard serving benchmark. ISL and concurrency are
properties of the request stream, not the connector:

```bash
python -m atom.benchmarks.benchmark_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 64 \
  --num-prompts 256 \
  --ignore-eos \
  --metric-percentiles 50,99 \
  --save-result --result-dir /tmp/bench_results \
  --result-filename partB_isl1024_conc64.json
```

- `--random-input-len` = ISL (the prompt length the sim allocates KV blocks for).
- `--max-concurrency` = in-flight requests (closed loop).
- `--random-output-len 1024 --ignore-eos` = fixed decode length (outputs are garbage,
  so EOS must be ignored to hit the target OSL).

To benchmark the decode server in isolation (no router, no producer), point
`--base-url` at `http://127.0.0.1:8020` directly — the consumer's `sim_autoprefill`
self-fabricates the skip-prefill path (Phase 1). This "pure-decode" path is the most
direct way to measure raw decode ceilings.

Reference baselines (tp=1, OSL 1024) captured on the MI355 node:

| Path | ISL/conc | Output tok/s | TTFT (mean) | TPOT (mean) |
|---|---|---|---|---|
| Router (Part B) | 1k/64 | 12,686 | 174 ms | 4.74 ms |
| Router (Part B) | 1k/512 | 28,615 | 1,032 ms | 16.24 ms |
| Router (Part B) | 8k/64 | 6,247 | 159 ms | 10.06 ms |
| Router (Part B) | 8k/512 | 7,640 | 8,022 ms | 52.14 ms (KV-capacity wall, eff. conc ≈457.8) |
| Pure-decode (:8020) | 1k/64 | ~12,686 | 74 ms | ~= router |

The 8k/512 point hits the KV-capacity wall: 8192×512 needs ~262,144 blocks but only
239,148 exist, so effective concurrency collapses to ~457.8. Adding decode TP relieves
this (see `sim_experiments_plan.md`, Experiment 1).

---

## 5. Proof-of-correctness logs

Two independent signals confirm the pipeline is doing real P/D control flow, not
falling back to local prefill.

**Consumer — skip-prefill actually happened.** The scheduler emits one
`[PD-TRANSITION]` line per request that completed the remote-KV path (full ISL
allocated, prefill forward skipped, flipped to `RUNNING`). The count of new lines
during a run should equal `--num-prompts`:

```bash
grep -c PD-TRANSITION /tmp/consumer.log
```

A sample line shows the full ISL was allocated with no prefill forward:

```
[PD-TRANSITION] seq <id>: num_tokens=..., num_prompt=8192, blocks=512, first_token=0, ...
```

**Producer — handshake payload was emitted.** On the producer, each finished request
logs that the KV-transfer output (the handshake payload the router relays to the
consumer) is present. Grep the producer log:

```bash
grep -iE "KV transfer output present|request_finished|kv_transfer_params_output" /tmp/producer.log
```

Together these prove: producer produced a handshake → router relayed it → consumer
consumed it and skipped prefill. No RDMA was involved; the block metadata traveled as
HTTP JSON.

---

## 6. Transfer-rate knobs (producer→decode latency emulation)

The consumer connector can pace per-request recv completion to emulate the
prefill→decode KV transfer latency. Two mutually-exclusive knobs (fixed delay wins):

| Config key | Meaning |
|---|---|
| `sim_transfer_delay_ms` | Fixed per-request delay in ms (takes priority). |
| `sim_transfer_gbps` | Bytes/s budget → `delay = KV_bytes(ISL) / rate`. |
| *(neither)* | Instant completion (default). |

Set them in the **consumer's** `--kv-transfer-config`, e.g.:

```bash
--kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"sim","sim_transfer_delay_ms":500}'
```

These shift **TTFT** (each request waits `ready_at` before flipping to RUNNING) and, in
a closed-loop benchmark, lower **output throughput** by Little's Law
(`throughput = concurrency / latency`). They do not affect TPOT. See
`sim_decode_connector_guide.md` §3 for the mechanics and `sim_experiments_plan.md`
Experiment 2 for measured effects.

---

## 7. Honest limitation: prefill TP/DP is NOT modeled

`--sim-prefill-tp` and `--sim-prefill-dp` exist as CLI args and are advertised as
"logical topology for handshake metadata," **but they are unused stubs**. In
`arg_utils.py::_get_engine_kwargs` they are popped with `# noqa: F841` and never wired
into anything:

```python
sim_prefill_tp = kwargs.pop("sim_prefill_tp", None)  # noqa: F841
sim_prefill_dp = kwargs.pop("sim_prefill_dp", None)  # noqa: F841
```

Consequently:

- The advertised prefill TP/DP topology is **not physically realized**. In
  compute-skip mode the producer's `world_size` is forced to **1** and device is CPU
  (`_setup_device_and_distributed_sim`), so there is no multi-rank prefill to model.
- The producer emits ~0 prefill latency regardless of these flags. If you need to model
  prefill TTFT contribution, use the consumer-side transfer-rate knobs (§6) instead,
  which is the only latency lever the simulator actually honors.
- **Decode** TP (`-tp N` on the consumer) is fully real and faithful — this limitation
  applies only to the *prefill* side.

Treat these two flags as reserved-for-future-use placeholders. Do not rely on them to
change any measured number today.

---

## 8. Cross-references

- Consumer / decode role, skip-prefill mechanics, KV-byte sharding under TP:
  [`sim_decode_connector_guide.md`](./sim_decode_connector_guide.md)
- Real (Mooncake/RDMA) P/D disaggregation the sim path mirrors:
  [`../recipes/pd_disaggregation_guide.md`](../recipes/pd_disaggregation_guide.md)
- Experiment recipes and results:
  [`sim_experiments_plan.md`](./sim_experiments_plan.md)
- Source: `atom/kv_transfer/disaggregation/sim/sim_connector.py`,
  `atom/kv_transfer/disaggregation/sim/sizing.py`,
  `atom/entrypoints/simulator_server.py`,
  `atom/model_engine/arg_utils.py` (sim fields), `atom/model_engine/model_runner.py`
  (`_setup_device_and_distributed_sim`).
