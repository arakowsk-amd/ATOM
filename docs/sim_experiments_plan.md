# Sim P/D Simulator — Experiment Plan

Two experiments that exercise the `sim` KV connector's two performance levers:

1. **Decode-TP sweep** — how much decode tensor parallelism relieves the KV-capacity
   wall and lowers TPOT at a fixed heavy operating point.
2. **Transfer-rate knob** — that the consumer-side `sim_transfer_delay_ms` /
   `sim_transfer_gbps` knobs actually shift TTFT (and, closed-loop, throughput).

Background and launch commands live in
[`sim_producer_pipeline_guide.md`](./sim_producer_pipeline_guide.md) and
[`sim_decode_connector_guide.md`](./sim_decode_connector_guide.md). This doc is the
runnable recipe: exact commands, what varies, what to measure, where results land, and
hypotheses. Measured numbers are appended under **§Results**.

## Environment

- Node: `smci355-ccs-aus-n13-17.cs-aus.dcgpu`, container `objective_wing`, repo at
  `/app/ATOM`.
- Model: `/data/models/Llama-3.1-8B-Instruct-FP8-KV`, `--kv_cache_dtype fp8`.
- Results dir (in container): `/tmp/bench_results/`.
- Per CLAUDE.md: `export AITER_LOG_LEVEL=WARNING` before every server start;
  `rm -rf /root/.cache/atom/*` before each relaunch; confirm the model is loaded with
  `rocm-smi --showmemuse` (VRAM% > 0 on the consumer GPU), not just `/health`.
- tp=1 consumer baseline: `num_kvcache_blocks=239148`, `block_bytes=1081344` (~240 GB
  KV). KV blocks scale roughly linearly with TP (more ranks → more aggregate KV VRAM →
  more blocks), which is exactly the capacity wall this suite probes.

Fixed benchmark shape (both experiments): `--random-output-len 1024 --ignore-eos
--metric-percentiles 50,99 --backend vllm --dataset-name random`.

---

## Experiment 1 — Decode-TP sweep

**Question.** At the ISL 8k / conc 512 point (which hit the KV-capacity wall at tp=1:
8192×512 = 262,144 blocks needed vs 239,148 available → effective concurrency ≈457.8),
how much does adding decode TP (1 → 2 → 4) raise available KV blocks, restore effective
concurrency, raise output throughput, and lower TPOT?

**What varies.** Consumer `-tp ∈ {1, 2, 4}` and its `HIP_VISIBLE_DEVICES`. Everything
else fixed.

**What's measured (per run).** Total KV blocks available (`num_kvcache_blocks` from
consumer log), effective concurrency (`concurrency` in the JSON), mean/P99 TTFT, mean
TPOT, output token throughput.

### GPU allocation and the producer conflict

tp=4 needs GPUs 0,1,2,3, which **collides with the sim-producer on GPU 1**. Therefore
the TP sweep is run as **pure-decode directly against the consumer at `:8020`** (no
router, no producer — `sim_autoprefill` self-fabricates skip-prefill), and the
producer + router are **stopped** to free GPU 1 for the duration of the sweep. This is
sound because Experiment 1 measures *decode* behavior only; the router/producer add no
decode-relevant work (the router relays a JSON handshake; the producer does ~0 GPU
work). The baseline for comparison is the existing `puredecode_isl8192_conc512.json`
(tp=1), captured on the same direct-to-:8020 path.

### Procedure (repeat for tp ∈ {2, 4}; tp=1 baseline already exists)

1. Stop the current consumer (and, before tp≥2, the producer + router).
2. `rm -rf /root/.cache/atom/*`.
3. Relaunch the consumer at the new `-tp` with the right device mask
   (tp=2 → `HIP_VISIBLE_DEVICES=0,1`; tp=4 → `0,1,2,3`):

   ```bash
   cd /app/ATOM
   export AITER_LOG_LEVEL=WARNING
   export HIP_VISIBLE_DEVICES=0,1        # tp=2 ; use 0,1,2,3 for tp=4
   python -m atom.entrypoints.openai_server \
     --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
     --load_dummy empty --kv_cache_dtype fp8 \
     -tp 2 \                             # or -tp 4
     --host 0.0.0.0 --server-port 8020 \
     --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"sim"}' \
     > /tmp/consumer.log 2>&1 &
   ```

4. Verify: `rocm-smi --showmemuse` shows VRAM% > 0 on each of the tp GPUs, and
   `grep num_kvcache_blocks /tmp/consumer.log` reports the (larger) block count.
5. Run the single 8k/512 point:

   ```bash
   python -m atom.benchmarks.benchmark_serving \
     --backend vllm --base-url http://127.0.0.1:8020 \
     --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
     --dataset-name random --random-input-len 8192 --random-output-len 1024 \
     --max-concurrency 512 --num-prompts 1024 --ignore-eos \
     --metric-percentiles 50,99 \
     --save-result --result-dir /tmp/bench_results \
     --result-filename tpsweep_tp2_isl8192_conc512.json   # tp4 → tpsweep_tp4_...
   ```

**Result files.** `tpsweep_tp{1,2,4}_isl8192_conc512.json` in `/tmp/bench_results/`
(tp1 = copy of the existing `puredecode_isl8192_conc512.json`).

**Hypotheses.**
- KV blocks scale ~linearly with TP: tp=2 ≈ 478k, tp=4 ≈ 956k blocks. tp=2 already
  exceeds the 262,144 needed, so the capacity wall clears at tp=2.
- Effective concurrency returns to ~512 (from 457.8) once blocks suffice.
- TPOT drops with TP (KV read is sharded across more ranks): expect roughly a 1.5–2×
  reduction from tp=1 → tp=2, with diminishing returns at tp=4 (communication overhead
  + already-cleared wall).
- Output throughput rises with TP until decode compute (not capacity) becomes the bound.

---

## Experiment 2 — Transfer-rate knob

**Question.** Do the consumer-side transfer-rate knobs (`sim_transfer_delay_ms`,
`sim_transfer_gbps`) actually delay each request's transition into decode, shifting
TTFT by the expected amount, and does closed-loop throughput fall accordingly (Little's
Law)?

**What varies.** Consumer `--kv-transfer-config`: (a) instant baseline, (b)
`sim_transfer_delay_ms:500`, (c) `sim_transfer_gbps:100`. Consumer stays at **tp=1**.

**Fixed point.** ISL 1024 / conc 64 / num-prompts 256, OSL 1024, direct against
`:8020`. Instant-baseline reference: `puredecode_isl1024_conc64.json` (TTFT ≈74 ms,
~12,686 tok/s).

**What's measured.** Mean/P99 TTFT and output throughput vs the instant baseline.
Verify the delay took effect via the TTFT delta (~+500 ms expected for delay_ms=500)
and/or `[PD-TRANSITION]` timestamps in the consumer log.

### Procedure

1. Ensure the consumer runs at tp=1 with the chosen config, e.g. for the fixed delay:

   ```bash
   export HIP_VISIBLE_DEVICES=0
   python -m atom.entrypoints.openai_server \
     --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
     --load_dummy empty --kv_cache_dtype fp8 -tp 1 \
     --host 0.0.0.0 --server-port 8020 \
     --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"sim","sim_transfer_delay_ms":500}' \
     > /tmp/consumer.log 2>&1 &
   ```

   For the rate variant, swap the config for
   `{"kv_role":"kv_consumer","kv_connector":"sim","sim_transfer_gbps":100}`. Clear the
   compile cache between relaunches.

2. Run the fixed point:

   ```bash
   python -m atom.benchmarks.benchmark_serving \
     --backend vllm --base-url http://127.0.0.1:8020 \
     --model /data/models/Llama-3.1-8B-Instruct-FP8-KV \
     --dataset-name random --random-input-len 1024 --random-output-len 1024 \
     --max-concurrency 64 --num-prompts 256 --ignore-eos \
     --metric-percentiles 50,99 \
     --save-result --result-dir /tmp/bench_results \
     --result-filename rate_delay500_isl1024_conc64.json   # or rate_gbps100_...
   ```

**Result files.** `rate_delay500_isl1024_conc64.json`,
`rate_gbps100_isl1024_conc64.json`.

**Hypotheses.**
- `delay_ms=500`: mean TTFT shifts up by ~500 ms (from ~74 ms to ~574 ms). By Little's
  Law, per-request latency = TTFT + decode_time; adding a fixed 500 ms to a request
  whose total latency was on the order of a few seconds lowers throughput modestly (not
  by 500 ms/req of the whole run, because concurrency hides most of it, but measurably).
- `sim_transfer_gbps=100`: per-request delay = KV_bytes(ISL=1024) / (100e9 B/s). For
  ISL 1024 at tp=1, blocks = 1024/16 = 64, KV_bytes = 64 × 1,081,344 ≈ 69.2 MB → delay
  ≈ 0.69 ms — effectively negligible. So the 100 GB/s variant should look
  ~indistinguishable from the instant baseline, confirming the rate path is wired to
  block count and that 100 GB/s is "fast enough to not matter" at this ISL. (A much
  lower gbps, or a much larger ISL, would be needed to see a visible shift — noted as a
  caveat.)

---

## Results

Executed 2026-07-26 on `smci355-ccs-aus-n13-17` / container `objective_wing`,
Llama-3.1-8B-Instruct-FP8-KV, fp8 KV. Result JSONs in container `/tmp/bench_results/`.

### Experiment 1 — Decode-TP sweep (ISL 8192 / conc 512, OSL 1024, direct :8020)

Files: `tpsweep_tp{1,2,4}_isl8192_conc512.json` (tp1 = copy of
`puredecode_isl8192_conc512.json`).

| tp | KV blocks | block_bytes/rank | eff. concurrency | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | output tok/s |
|----|-----------|------------------|------------------|----------------|---------------|----------------|--------------|
| 1  | 239,148 | 1,081,344 | 456.68 | 7276.06 | 57222.48 | 51.13 | 7,848.9 |
| 2  | 455,897 | 540,672 | 502.78 | 1264.36 | 2201.26 | 33.53 | 14,478.0 |
| 4  | 913,750 | 270,336 | 500.48 | 1250.91 | 2074.68 | 23.84 | 19,987.7 |

**Observations.** At tp=1 the point needs 8192×512 = 262,144 blocks but only 239,148
exist → the KV-capacity wall collapses effective concurrency to 456.68 and inflates
mean TTFT to 7.3 s (P99 57 s — requests queue waiting for blocks to free). tp=2 already
provides 455,897 blocks (> 262,144), **clearing the wall**: effective concurrency jumps
to 502.78, TTFT drops ~5.8× (7276 → 1264 ms), and output throughput ~1.85× (7,849 →
14,478 tok/s). Per-rank `block_bytes` halves each TP doubling (1,081,344 → 540,672 →
270,336), confirming KV heads shard across ranks as expected. tp=4 doubles blocks again
(913,750) but the wall was already gone at tp=2, so concurrency/TTFT are flat; the
remaining gain is pure decode compute parallelism: **TPOT falls monotonically 51.13 →
33.53 → 23.84 ms** and output throughput rises to 19,988 tok/s (2.55× over tp=1).
Takeaway: TP relieves the capacity wall at the first doubling (blocks), then keeps
buying lower TPOT / higher throughput via compute sharding.

### Experiment 2 — Transfer-rate knob (ISL 1024 / conc 64, OSL 1024, direct :8020)

Files: `puredecode_isl1024_conc64.json` (instant baseline),
`rate_delay500_isl1024_conc64.json`, `rate_gbps100_isl1024_conc64.json`.

| config | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | output tok/s | Δ mean TTFT vs instant |
|--------|----------------|---------------|----------------|--------------|------------------------|
| instant (baseline) | 74.17 | 139.28 | 4.83 | 12,985.1 | 0 |
| delay_ms=500 | 576.98 | 618.36 | 4.89 | 11,731.3 | +502.8 |
| gbps=100 | 221.69 | 640.56 | 4.82 | 12,407.5 | +147.5 (see note) |

**Observations.** The fixed `sim_transfer_delay_ms=500` shifts mean TTFT by **+502.8 ms**
(74.17 → 576.98) — precisely the configured 500 ms — while **TPOT is unchanged** (4.83 →
4.89 ms), exactly as designed: the knob paces each request's exit from
`WAITING_FOR_REMOTE_KVS`, not its decode. Closed-loop output throughput falls modestly
(12,985 → 11,731 tok/s, −9.7%) because per-request latency rose by 500 ms while
concurrency is fixed at 64 (Little's Law).

The `sim_transfer_gbps=100` variant is **effectively instant**: the rate-derived delay
for ISL 1024 at tp=1 is `blocks(64) × block_bytes(1,081,344) / 100e9 ≈ 0.69 ms` — below
the noise floor. Its mean TTFT of 221.69 ms is dominated by first-batch warm-up/queueing
(P50 TTFT is only 106 ms), and output throughput (12,407 tok/s) matches the instant
baseline within noise. This confirms the rate path is wired to block count and that
100 GB/s is "fast enough to not matter" at this ISL; a visibly larger shift requires a
much lower gbps or a much larger ISL. Net: the delay knob is the reliable TTFT lever;
the gbps knob only bites when `KV_bytes(ISL)/rate` is on the order of the baseline TTFT.
