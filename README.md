# Edge Inference Systems Engineering

Hands-on experiments in **local LLM inference performance, profiling, and evaluation** on resource-constrained hardware. The goal is to turn performance observations into reproducible measurements, testable hypotheses, and defensible engineering decisions.

This repository develops a small Python-based inference benchmarking toolkit, starting with Ollama on a **Raspberry Pi 5 (8 GB)**. Experiments prioritise understanding system behaviour before optimising it.

## Modules

| Module | Focus | Experiments |
| --- | --- | --- |
| [01 — Inference measurement fundamentals](module_01/README.md) | Latency, streaming, prompt caching, prefill scaling, and thermal validation | [`benchmark.py`](module_01/benchmark.py), [`benchmark_stream.py`](module_01/benchmark_stream.py), [`caching.py`](module_01/caching.py), [`prefill_scaling.py`](module_01/prefill_scaling.py) |

Future work will expand into Linux CPU profiling, memory and cache analysis, concurrency, energy efficiency, inference-runtime tuning, and evaluation/regression testing.

## Selected findings

- **Cold-start cost matters:** In one experiment, model loading took **18.192 s** on the first request but only **0.001 s** on the next. Client-observed latency dropped from **24.239 s** to **6.091 s**.
- **Streaming changes how latency is interpreted:** A warm request produced its first visible text at **0.385 s**, while the full response took **5.973 s**. Time to first visible text, full-response latency, and generation throughput answer different questions.
- **Caching is model/runtime-dependent:** On the tested Ollama setup, `qwen3.5:2b` reused **293/297** prompt tokens for repeated requests, while `lfm3-vl:2b` reported no cached tokens in the corresponding test. That does **not** establish that LFM3-VL fundamentally lacks caching support.
- **Prefill scaling is workload-dependent:** Uncached prefill throughput for `qwen3.5:2b` was approximately **39 tokens/s** up to ~2.7k input tokens, falling to around **37.7 tokens/s** near 3.6k tokens.
- **Measure before diagnosing throttling:** Hardware monitoring during longer runs showed temperatures up to **73.6°C**, an active ARM clock near **2.4 GHz**, and `throttled=0x0` across the recorded samples—no observed evidence of thermal throttling.

These are **device-, model-, configuration-, and workload-specific observations**, not universal model benchmarks. See [Module 01](module_01/README.md) for methods and limitations.

## Quick start

**Requirements:** Linux or another Ollama-supported environment, Python 3, a running [Ollama](https://ollama.com/) server, and at least one local model.

```bash
ollama list
ollama pull qwen3.5:2b  # if not already installed
python3 module_01/benchmark.py
python3 module_01/benchmark_stream.py
python3 module_01/caching.py
python3 module_01/prefill_scaling.py
```

Check the model name and experiment parameters inside each script before running. Larger prompts and cold starts can take substantial time on a Raspberry Pi.

## Benchmarking principles

- Record **model, runtime, hardware, prompt/token counts, cache hits, and configuration** alongside timings.
- Separate **client end-to-end latency** from **inference-runtime stage timings**.
- Compare cold and warm runs explicitly; don't mix them into one latency estimate.
- Repeat measurements, change run order, and monitor hardware before attributing a slowdown to a bottleneck.
- Treat performance explanations as **hypotheses to test**, not conclusions from a single timing result.

## References

- [Ollama API documentation](https://docs.ollama.com/api)
- [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/)
