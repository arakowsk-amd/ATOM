# Sim-Decode Connector — Usage Guide

The `sim` KV connector is a **compute-free stand-in for a real prefill/decode (P/D)
transfer**. In **consumer / decode** role it lets a *real* decode server (real weights,
real GPU decode compute) run **without a prefill server, router, or RDMA**.

For every incoming request the connector forces the "remotely matched" path, so the
scheduler:

1. parks the request in `WAITING_FOR_REMOTE_KVS`,
2. allocates the full input-sequence-length (ISL) of KV blocks,
3. **skips the prefill forward pass**, and
4. injects a synthetic first token and flips the request to `RUNNING` — real decode
   compute then runs over the (garbage) KV blocks.

Decode performance is **content-independent** (paged attention reads every ISL block
from the block table regardless of contents), so TPOT / ITL / output-throughput are
faithful even though the KV bytes are never really produced.

---

## 1. Launch a sim-decode server

Select the sim consumer connector via `--kv-transfer-config`. Everything else is a
normal ATOM decode server (use your own real `-tp` / `-dp`).

```bash
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model amd/Llama-3.1-8B-Instruct-FP8-KV \
  --kv-cache-dtype fp8 -tp 8 \
  --max-num-seqs 4096 \
  --server-port 8011 \
  --kv-transfer-config '{"kv_connector":"sim","kv_role":"kv_consumer"}'
```

Notes:
- `-tp` / `-dp` are the decode server's own **real** topology. The sim consumer shards
  its KV byte accounting by the real `tp_size` automatically.
- `--max-num-seqs` caps the real decode batch. If you want to probe the true decode
  ceiling, set it high enough that client concurrency — not this cap — is the limit
  (e.g. `4096`). Leaving it at the default (`512`) caps running requests at 512 and
  queues the rest.

---

## 2. Drive load with the benchmark client

ISL and concurrency are properties of the **request stream**, not the connector:

- **ISL** = each request's prompt length → `--random-input-len <ISL>`
- **Concurrency** = requests in flight → `--max-concurrency <C>`

```bash
python -m atom.benchmarks.benchmark_serving \
  --model amd/Llama-3.1-8B-Instruct-FP8-KV \
  --backend vllm \
  --base-url http://127.0.0.1:8011 \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --max-concurrency 128 \
  --num-prompts 1280
```

Outputs are garbage text (expected). The metrics (TTFT, TPOT, ITL, throughput) are
faithful.

---

## 3. Transfer-rate knobs (emulate prefill→decode transfer latency)

By default recv completes **instantly**. Two optional knobs pace per-request
completion, which staggers when each request leaves `WAITING_FOR_REMOTE_KVS` and enters
`RUNNING` — a TTFT-like delay, and a cap on how fast new decodes can start.

| Config key                | Meaning                                            |
|---------------------------|----------------------------------------------------|
| `sim_transfer_delay_ms`   | **Fixed** per-request delay in ms (takes priority) |
| `sim_transfer_gbps`       | Bytes/s budget → delay = `KV_bytes(ISL) / rate`    |
| *(neither set)*           | Instant completion (default)                       |

Per-request KV bytes are derived weight-free from the request's block count and the HF
config (sharded by `tp_size`).

**Fixed 500 ms delay per request:**

```bash
--kv-transfer-config '{"kv_connector":"sim","kv_role":"kv_consumer","sim_transfer_delay_ms":500}'
```

**1 GB/s transfer-rate budget (TTFT scales with ISL):**

```bash
--kv-transfer-config '{"kv_connector":"sim","kv_role":"kv_consumer","sim_transfer_gbps":1.0}'
```

These knobs affect **TTFT and throughput**, not TPOT — completion is modeled as an
independent per-stream latency, not a shared link. In a closed-loop
(fixed-concurrency) benchmark, added TTFT raises per-request latency, so by Little's
Law (`throughput = concurrency / latency`) output-throughput falls.

---

## 4. Other config keys

| Config key            | Default          | Meaning                                              |
|-----------------------|------------------|------------------------------------------------------|
| `kv_connector`        | —                | Must be `"sim"` to select this backend               |
| `kv_role`             | `kv_producer`    | Use `"kv_consumer"` for the decode-sim role          |
| `sim_autoprefill`     | `true` (consumer)| Force every request onto the skip-prefill path       |
| `sim_first_token_id`  | `0`              | Synthetic first token injected on transfer completion|
| `sim_transfer_delay_ms` | *(unset)*      | Fixed per-request completion delay (ms)              |
| `sim_transfer_gbps`   | *(unset)*        | Transfer-rate budget (GB/s) → derived delay          |

---

## 5. Confirming prefill is actually skipped

The scheduler emits a `[PD-TRANSITION]` log line **only** when a request completes the
remote-KV path (i.e. skipped prefill and was flipped to `RUNNING`). Counting these
lines is airtight proof:

```bash
# new PD-TRANSITION lines during a run should equal --num-prompts
grep -c PD-TRANSITION <server-log>
```

A sample line shows the full ISL was allocated with no prefill forward:

```
[PD-TRANSITION] seq <id>: num_tokens=..., num_prompt=4096, blocks=256, first_token=0, ...
```

---

## 6. Multi-GPU (TP > 1)

Just set the decode server's real `-tp`. The sim consumer reads the real
`get_tp_group().world_size` and shards its per-rank KV byte accounting accordingly
(e.g. `tp=8` → KV block bytes / 8 per rank). Skip-prefill behavior is unchanged.
