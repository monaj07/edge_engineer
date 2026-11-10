# Module 01 — Inference Measurement Fundamentals

**Objective:** Learn to measure local LLM inference accurately, understand the main latency components, detect prompt-cache reuse, investigate prefill scaling, and test hardware-related explanations for slowdowns.

**Testbed:** Raspberry Pi 5 (8 GB), Linux, Python, and an Ollama server. Models tested include `qwen3.5:2b` and `lfm3-vl:2b`. All figures below are observations from this setup, **not cross-device or cross-model performance claims**.

## Experiments and scripts

| Script | Focus |
| --- | --- |
| [`benchmark.py`](benchmark.py) | Compare client wall-clock time with Ollama's load, prefill, and generation timings; observe cold vs. warm requests |
| [`benchmark_stream.py`](benchmark_stream.py) | Measure first visible text, streaming chunk arrival, and end-to-end latency |
| [`caching.py`](caching.py) | Compare identical prompts and prompts changed at the beginning/end; inspect reported cache hits |
| [`prefill_scaling.py`](prefill_scaling.py) | Vary uncached input length and measure prefill duration and throughput |

### Run locally

Start Ollama, check which models are installed, and select the model in each script:

```bash
ollama list
# Optional: unload a model before a cold-start test
ollama stop qwen3.5:2b

python3 module_01/benchmark.py
python3 module_01/benchmark_stream.py
python3 module_01/caching.py
python3 module_01/prefill_scaling.py
```

Commands assume execution from the repository root. Scripts use the Ollama HTTP API on `localhost:11434`. Confirm that `MODEL`, context size, token budget, and prompt construction match the experiment you intend to reproduce. For a fair comparison, record the actual returned token counts rather than assuming text length equals token count.

## 1. What are we measuring?

A request can spend time in several distinct stages:

1. **Load:** Make the model available to the inference runtime (cold start vs. already resident).
2. **Prefill / prompt evaluation:** Process input tokens and initialise/update inference state.
3. **Decode / output generation:** Produce output tokens autoregressively.
4. **Client overhead:** HTTP request/response handling, transport and JSON processing, among other costs.

Ollama reports `total_duration`, `load_duration`, `prompt_eval_duration`, and `eval_duration` in **nanoseconds**; divide by `1e9` for seconds. Token counts (`prompt_eval_count`, `eval_count`) are essential when comparing different workloads. Client wall time (e.g. `time.perf_counter()`) includes work outside the server's timing boundary, so it is not necessarily identical to `total_duration`.

**Useful metrics:**

| Metric | Meaning / calculation |
| --- | --- |
| End-to-end latency | Client request start → completed response |
| Time to first visible text (TTFT-like) | Client request start → first non-empty streamed `response` chunk |
| Time to usable chunk | Request start → enough visible text for a downstream task, such as TTS |
| Decode throughput | `eval_count / (eval_duration / 1e9)` in tokens/s |
| Effective prefill throughput | `prompt_eval_count / (prompt_eval_duration / 1e9)` in tokens/s for uncached prompts |
| Incremental prefill throughput | `(N₂ − N₁) / (T₂ − T₁)` between two input lengths |

**Streaming caveat:** An Ollama text chunk is not guaranteed to represent exactly one token. The 20th *text chunk* is therefore not necessarily the 20th token. Thinking-capable models may emit content under `thinking` rather than `response`; using `"think": false` (when supported) makes time-to-first-visible-answer measurements easier to interpret. `num_predict` is also a generation budget; a model may stop before reaching it.

## 2. Cold versus warm inference

For a 20-input-token / 60-output-token request:

| Metric | First request | Subsequent request |
| --- | ---: | ---: |
| Client wall time | 24.239 s | 6.091 s |
| Ollama total duration | 24.218 s | 6.090 s |
| Model loading | **18.192 s** | **0.001 s** |
| Prompt evaluation | 0.361 s | 0.360 s |
| Token generation | 5.663 s | 5.726 s |

**Finding:** The difference is almost entirely explained by the reported model-load time. Prefill and generation are similar; repeated end-to-end measurements alone would not have identified the cause.

For another streaming workload (59 output tokens), cold vs. warm results were:

| Metric | Cold | Warm |
| --- | ---: | ---: |
| First visible text | 3.757 s | **0.385 s** |
| 20th visible text chunk | 5.594 s | 2.224 s |
| End-to-end latency | 9.394 s | **5.973 s** |
| Load duration | 3.377 s | 0.001 s |
| Generation duration | 5.636 s | 5.595 s |

In the warm run, generation accounted for **~94% of total client latency**. Loading affects responsiveness, but the steady-state bottleneck was output generation for this short-input workload. Short TTFT is not the same as short total latency; for voice systems, *time to first usable TTS chunk* can be more relevant than either.

## 3. Prompt caching: never assume a cache hit

Prefix caching can reuse work for the initial sequence of unchanged tokens. Modifying a suffix may leave a useful common prefix; modifying the beginning usually prevents reuse of user-prompt tokens. Templates, runtime state, and multiple retained cache entries complicate the picture.

**Identical 297-token requests:**

| Model | Request 1 | Requests 2–3 |
| --- | --- | --- |
| `lfm3-vl:2b` | 0 cached; 4.379 s prefill | 0 cached; ~4.33 s prefill |
| `qwen3.5:2b` | 0 cached; 7.532 s prefill | **293 cached; 0.224 s prefill** |

**Controlled Qwen test (`raw=True`, 285 tokens):**

| Request | First sequence: cached / prefill | Repeated sequence: cached / prefill |
| --- | --- | --- |
| Original | 0 / 6.992 s | 281 / 0.210 s |
| Changed end | 281 / 0.212 s | 281 / 0.251 s |
| Changed start | 0 / 6.850 s | 281 / 0.211 s |

The repeated sequence suggests that distinct previously evaluated prompt states can be reused. `lfm3-vl:2b` not showing cache hits **does not prove caching is fundamentally unsupported**; it may be runtime-, configuration-, or model-implementation-dependent.

**Method:** Keep the model loaded, compare identical/modified requests, record `prompt_eval_cached_count`, and use `raw=True` when testing literal prompt prefixes. A unique identifier at the *beginning* of each benchmark prompt reduces unwanted prefix reuse; always verify cached-token counts rather than assuming it worked.

## 4. Prefill scaling with input length

With `qwen3.5:2b`, `think=False`, minimal output, fixed context configuration, and **zero reported cached tokens**, three repeated measurements yielded:

| Mean input tokens | Mean prefill duration | Approx. throughput |
| ---: | ---: | ---: |
| 57 | 1.518 s | 37.8 tok/s |
| 143 | 3.513 s | 40.7 tok/s |
| 480 | 11.533 s | 41.6 tok/s |
| 927 | 22.540 s | 41.1 tok/s |

An initial empirical slope between two observations was:

`(22.731 − 3.587) / (929 − 143) ≈ 0.02436 s/token`

Its reciprocal, **~41.06 tokens/s**, is the *incremental* prefill throughput between those points; it is not the slope in seconds/token. A linear extrapolation predicted ~**17.2 s for 700 tokens** (not a guarantee outside tested conditions).

Extending to longer prompts gave:

| Input tokens | Prefill | Effective throughput |
| ---: | ---: | ---: |
| 925 | 23.479 s | 39.40 tok/s |
| 1,822 | 46.599 s | 39.10 tok/s |
| 2,717 | 68.906 s | 39.43 tok/s |
| 3,615 | 95.924 s | 37.69 tok/s |

The first three points are nearly constant in throughput; the longest prompt is somewhat slower. Incremental throughput decreased from **~40.12 tok/s** (1,822 → 2,717) to **~33.24 tok/s** (2,717 → 3,615). One observation per size cannot identify the cause.

### Why not obviously quadratic?

For conventional full causal attention, token interactions scale as `O(N²)`, but projections and feed-forward layers scale roughly as `O(N)` for fixed hidden dimensions. A useful simplified workload model is:

`T(N) ≈ c + aN + bN²`

Qwen3.5 additionally uses a **hybrid linear/full-attention architecture**, reducing the share of quadratic-attention computation. Short-to-moderate context sizes may therefore exhibit approximately linear elapsed time. **Algorithmic FLOPs complexity is not the same as measured wall-clock scaling**: utilisation, memory hierarchy, and runtime implementation also matter. These measurements do not prove a quadratic crossover.

## 5. Check the hardware before blaming thermal throttling

We counterbalanced prompt lengths using an **ABBA** order (~2.7k, ~3.6k, ~3.6k, ~2.7k tokens), then repeated the sequence. All eight runs reported zero cached tokens.

| Prompt length | Runs | Mean throughput |
| --- | ---: | ---: |
| ~2,716 tokens | 4 | **38.70 tok/s** |
| ~3,614 tokens | 4 | **37.73 tok/s** |

The longer prompts were ~2.5% slower in mean effective throughput. Raspberry Pi monitoring during the second sequence recorded:

- **Temperature:** up to 73.6°C.
- **ARM clock during active inference:** approximately 2.4 GHz; lower readings around 1.5 GHz occurred during idle/cool-down periods.
- **Throttling flags:** `throttled=0x0` throughout the provided monitoring log.

**Interpretation:** There was **no observed evidence of thermal or power throttling** in this monitoring window. That weakens throttling as the explanation for the throughput difference; it does **not** prove quadratic attention is the cause.

Example monitoring command (run in a separate terminal):

```bash
while true; do
  date '+%H:%M:%S'
  vcgencmd measure_temp
  vcgencmd measure_clock arm
  vcgencmd get_throttled
  sleep 2
done | tee /tmp/pi_thermal.log
```

For more rigorous comparisons, record per-request start/end timestamps to align hardware data with the corresponding workload; sample frequently enough to detect short changes and retain raw logs.

## 6. Measurement lessons and next steps

- **Isolate phases:** model loading, prompt evaluation, decoding and client overhead should not be conflated.
- **Control cache state:** record cached-token counts for every run. Stopping a model may not clear OS file caches.
- **Repeat and counterbalance:** A/B comparisons need repeated samples and order control; a single result cannot establish a bottleneck.
- **Measure rather than speculate:** temperature alone isn't proof of throttling; inspect clock behaviour and throttle flags.
- **Separate observations from explanations:** modest throughput degradation is real in these samples; its microarchitectural cause remains to be investigated.

**Next:** Linux performance profiling: CPU cycles, instructions per cycle (IPC), cache misses, memory bandwidth, and repeatable performance diagnostics.

### References

- [Ollama API — Generate a response](https://docs.ollama.com/api/generate)
- [Ollama API — Streaming](https://docs.ollama.com/api/streaming)
- [Raspberry Pi documentation — `vcgencmd`](https://www.raspberrypi.com/documentation/computers/os.html)
