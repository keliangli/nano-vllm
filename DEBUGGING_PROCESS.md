# Debugging Process

## Goal

Set up a stable local `nano-vllm` environment for the local Qwen3-0.6B model, measure inference speed on an RTX 3070 8GB GPU, and progressively optimize the runtime path.

## Environment

- GPU: NVIDIA GeForce RTX 3070 8GB
- Driver / CUDA runtime: 550.78 / CUDA 12.4
- Python env used for final validation: `mini-vllm-clean`
- Core packages:
  - `torch 2.5.1+cu124`
  - `triton 3.1.0`
  - `transformers 4.51.0`
  - `flashinfer-python 0.6.6`

## Setup Notes

- Switched `pip` to Tsinghua mirror for faster dependency installation.
- Final working env was created with Python 3.10 and a clean dependency stack rather than reusing the existing mixed conda environment.

## Issues Found

### 1. Wrong backend / missing optimized kernels

Initially the runtime did not consistently use FlashInfer and sometimes fell back to the pure PyTorch attention path, causing very low throughput.

### 2. FlashInfer runtime incompatibility

Early attempts with the preinstalled environment hit:

- missing `ninja`
- `libstdc++` ABI mismatch
- FlashInfer JIT / segfault issues

This was solved by moving to a clean environment and installing a matching stack for `torch 2.5.1 + cu124`.

### 3. Default KV-cache sizing too aggressive for 8GB VRAM

With default config, KV-cache allocation failed on the 3070 8GB. Stable testing used:

- `max_model_len=1024`
- `max_num_batched_tokens=1024`
- `max_num_seqs=2`
- `gpu_memory_utilization=0.98`

### 4. FlashInfer decode path was not actually using the correct decode kernel

The previous implementation called the wrong FlashInfer path for decode and also passed query tensors with the wrong shape.

### 5. Python loops dominated runtime

Two major bottlenecks were found in attention:

- KV-cache writes were done token-by-token in Python
- KV-cache reads rebuilt the sequence token-by-token in Python

These loops erased much of the benefit from FlashInfer.

### 6. CUDA graph capture failed

`enforce_eager=False` initially failed because:

- KV-cache writes contained host-side branching on CUDA tensors
- decode logic still used Python-side scalar extraction during graph capture

## Code Changes

### Backend controls

Added env-based switches for:

- disabling FlashInfer / FlashAttention fallback selection
- disabling `torch.compile` for the sampler
- disabling batched FlashInfer wrappers when needed

### Attention path

In `nanovllm/layers/attention.py`:

- fixed FlashInfer decode API usage
- fixed decode query tensor shape
- vectorized KV-cache write path with `index_copy_`
- vectorized KV-cache read path with tensor indexing
- added a graph-capture-safe decode path
- added batched FlashInfer wrapper execution path

### Model runner

In `nanovllm/engine/model_runner.py`:

- added FlashInfer wrapper creation and planning
- wired paged KV-cache metadata into batched prefill/decode wrapper calls
- made eager / batched / cudagraph runtime modes separable
- disabled batched wrapper allocation in cudagraph mode to avoid OOM on 8GB VRAM

### Engine lifecycle

In `nanovllm/engine/llm_engine.py`:

- made `exit()` idempotent so benchmark scripts can call it safely without `atexit` noise

### Benchmarking

Added `benchmark_speed.py` to the repo root with default nano-vLLM mode comparison:

- `eager`
- `batched`
- `cudagraph`

The script now writes structured JSON results and prints side-by-side comparisons.

## Benchmark Evolution

### 1. Pure fallback / early state

The fallback-like path was very slow, with output throughput around:

- `2.47 tok/s` at `(128, 32, 1)`
- `2.92 tok/s` at `(128, 32, 2)`
- `0.88 tok/s` at `(512, 32, 1)`

### 2. FlashInfer fixed + KV vectorized

After correcting FlashInfer decode and vectorizing KV access:

- `(128, 32, 1)`: `10.43 tok/s`
- `(128, 32, 2)`: `17.60 tok/s`
- `(512, 32, 1)`: `10.46 tok/s`

### 3. Batched FlashInfer wrapper

After moving prefill/decode to batched paged-KV wrappers:

- `(128, 32, 1)`: `12.25 tok/s`
- `(128, 32, 2)`: `24.29 tok/s`
- `(512, 32, 1)`: `12.37 tok/s`

### 4. CUDA graph mode

After making KV writes graph-safe and adding a capture-safe decode path:

- `(128, 32, 1)`: `83.88 tok/s`
- `(128, 32, 2)`: `108.09 tok/s`
- `(256, 32, 1)`: `84.54 tok/s`
- `(256, 32, 2)`: `107.79 tok/s`
- `(512, 32, 1)`: `83.47 tok/s`

## Final Mode Comparison

Results from `benchmark_speed.py --engine nano-vllm --nano-mode all --profile default`:

| Input | Output | Concurrency | Eager out tok/s | Batched out tok/s | CUDAGraph out tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| 128 | 32 | 1 | 11.22 | 14.25 | 83.88 |
| 128 | 32 | 2 | 18.31 | 29.30 | 108.09 |
| 256 | 32 | 1 | 11.72 | 14.86 | 84.54 |
| 256 | 32 | 2 | 18.53 | 28.37 | 107.79 |
| 512 | 32 | 1 | 11.35 | 14.47 | 83.47 |

## Conclusions

- On this 8GB card, `cudagraph` is the strongest stable path.
- `batched` improves over `eager`, especially at concurrency 2.
- The biggest gains came from removing Python loops and making decode graph-safe.
- Full graph-friendly prefill is still improvable, but the current setup is already a large step up from the original path.
