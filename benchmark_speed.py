#!/usr/bin/env python3
"""
Benchmark vLLM and nano-vLLM on local models.

For nano-vLLM, the default benchmark compares three execution modes:
- eager: single-kernel FlashInfer path, no batched wrapper
- batched: batched FlashInfer wrapper path
- cudagraph: CUDA-graph decode path (enforce_eager=False)
"""

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from random import randint, seed
from typing import Dict, List, Optional, Tuple

import torch

# Keep sampler compile disabled during benchmarking to avoid autotune/OOM noise.
os.environ.setdefault("NANOVLLM_DISABLE_TORCH_COMPILE", "1")


@dataclass
class BenchmarkResult:
    engine: str
    mode: str
    input_len: int
    output_len: int
    concurrency: int
    output_tokens: int
    total_tokens: int
    time_seconds: float
    throughput_tokens_per_sec: float
    total_tokens_per_sec: float
    time_to_first_token_ms: Optional[float] = None


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def default_configs(profile: str) -> List[Tuple[int, int, int]]:
    if profile == "full":
        return [
            (128, 32, 1),
            (128, 32, 2),
            (256, 32, 1),
            (256, 32, 2),
            (512, 32, 1),
            (512, 32, 2),
            (768, 32, 1),
        ]
    return [
        (128, 32, 1),
        (128, 32, 2),
        (256, 32, 1),
        (256, 32, 2),
        (512, 32, 1),
    ]


def generate_random_token_ids(num_tokens: int, vocab_ceiling: int = 10000) -> List[int]:
    return [randint(0, vocab_ceiling) for _ in range(num_tokens)]


def run_vllm_benchmarks(
    model_path: str,
    configs: List[Tuple[int, int, int]],
    max_model_len: int,
    gpu_memory_utilization: float,
) -> List[BenchmarkResult]:
    from vllm import LLM, SamplingParams

    print("Loading vLLM model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=min(gpu_memory_utilization, 0.9),
        enforce_eager=True,
        dtype="float16",
    )

    print("Warming up...")
    llm.generate(
        [dict(prompt_token_ids=[1] * 32)],
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )

    results: List[BenchmarkResult] = []
    seed(42)

    for input_len, output_len, concurrency in configs:
        print(
            f"Testing vLLM: input={input_len:4d}, output={output_len:4d}, concurrency={concurrency:3d} ... ",
            end="",
            flush=True,
        )
        try:
            prompts = [
                dict(prompt_token_ids=generate_random_token_ids(input_len))
                for _ in range(concurrency)
            ]
            params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
            torch.cuda.synchronize()
            start = time.time()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            ttft_ms = None
            if outputs and hasattr(outputs[0], "metrics"):
                metrics = outputs[0].metrics
                if hasattr(metrics, "time_to_first_token") and metrics.time_to_first_token is not None:
                    ttft_ms = round(metrics.time_to_first_token * 1000, 2)

            output_tokens = output_len * concurrency
            total_tokens = (input_len + output_len) * concurrency
            result = BenchmarkResult(
                engine="vllm",
                mode="eager",
                input_len=input_len,
                output_len=output_len,
                concurrency=concurrency,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                time_seconds=round(elapsed, 3),
                throughput_tokens_per_sec=round(output_tokens / elapsed, 2),
                total_tokens_per_sec=round(total_tokens / elapsed, 2),
                time_to_first_token_ms=ttft_ms,
            )
            results.append(result)
            print(
                f"OK | {elapsed:6.2f}s | {result.throughput_tokens_per_sec:8.2f} out tok/s | "
                f"{result.total_tokens_per_sec:8.2f} total tok/s"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    del llm
    cleanup_gpu()
    return results


def configure_nano_env(mode: str) -> bool:
    os.environ["NANOVLLM_DISABLE_TORCH_COMPILE"] = "1"
    if mode == "eager":
        os.environ["NANOVLLM_DISABLE_FLASHINFER_BATCHED"] = "1"
        return True
    if mode == "batched":
        os.environ["NANOVLLM_DISABLE_FLASHINFER_BATCHED"] = "0"
        return True
    if mode == "cudagraph":
        os.environ["NANOVLLM_DISABLE_FLASHINFER_BATCHED"] = "1"
        return False
    raise ValueError(f"Unsupported nano mode: {mode}")


def run_nano_vllm_mode(
    model_path: str,
    configs: List[Tuple[int, int, int]],
    mode: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> List[BenchmarkResult]:
    from nanovllm import LLM, SamplingParams

    enforce_eager = configure_nano_env(mode)
    max_num_seqs = max(concurrency for _, _, concurrency in configs)
    max_num_batched_tokens = max_model_len

    print(f"Loading nano-vLLM model in mode={mode}...")
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print("Warming up...")
    llm.generate(
        [[1] * 32],
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )

    results: List[BenchmarkResult] = []
    seed(42)

    for input_len, output_len, concurrency in configs:
        print(
            f"Testing nano-vLLM({mode}): input={input_len:4d}, output={output_len:4d}, concurrency={concurrency:3d} ... ",
            end="",
            flush=True,
        )
        try:
            prompts = [generate_random_token_ids(input_len) for _ in range(concurrency)]
            params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
            torch.cuda.synchronize()
            start = time.time()
            llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            output_tokens = output_len * concurrency
            total_tokens = (input_len + output_len) * concurrency
            result = BenchmarkResult(
                engine="nano-vllm",
                mode=mode,
                input_len=input_len,
                output_len=output_len,
                concurrency=concurrency,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                time_seconds=round(elapsed, 3),
                throughput_tokens_per_sec=round(output_tokens / elapsed, 2),
                total_tokens_per_sec=round(total_tokens / elapsed, 2),
            )
            results.append(result)
            print(
                f"OK | {elapsed:6.2f}s | {result.throughput_tokens_per_sec:8.2f} out tok/s | "
                f"{result.total_tokens_per_sec:8.2f} total tok/s"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            import traceback

            traceback.print_exc()

    llm.exit()
    del llm
    cleanup_gpu()
    return results


def run_nano_vllm_benchmarks(
    model_path: str,
    configs: List[Tuple[int, int, int]],
    nano_mode: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Dict[str, List[BenchmarkResult]]:
    modes = ["eager", "batched", "cudagraph"] if nano_mode == "all" else [nano_mode]
    results_by_mode: Dict[str, List[BenchmarkResult]] = {}
    for mode in modes:
        cleanup_gpu()
        results_by_mode[mode] = run_nano_vllm_mode(
            model_path=model_path,
            configs=configs,
            mode=mode,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return results_by_mode


def print_summary(results: List[BenchmarkResult], title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("-" * 110)
    print(
        f"{'Mode':>12} {'Input':>8} {'Output':>8} {'Conc':>6} {'Time(s)':>10} "
        f"{'Out tok/s':>12} {'Total tok/s':>14} {'TTFT(ms)':>12}"
    )
    print("-" * 110)
    for r in results:
        ttft = f"{r.time_to_first_token_ms:>10.2f}" if r.time_to_first_token_ms is not None else "N/A"
        print(
            f"{r.mode:>12} {r.input_len:>8} {r.output_len:>8} {r.concurrency:>6} "
            f"{r.time_seconds:>10.2f} {r.throughput_tokens_per_sec:>12.2f} {r.total_tokens_per_sec:>14.2f} {ttft:>12}"
        )
    print("-" * 110)


def print_nano_mode_comparison(results_by_mode: Dict[str, List[BenchmarkResult]]) -> None:
    if len(results_by_mode) < 2:
        return

    print("\n" + "=" * 110)
    print("nano-vLLM Mode Comparison")
    print("-" * 110)

    mode_order = ["eager", "batched", "cudagraph"]
    available_modes = [mode for mode in mode_order if mode in results_by_mode]
    base_mode = available_modes[0]
    key_order = [
        (r.input_len, r.output_len, r.concurrency)
        for r in results_by_mode[base_mode]
    ]

    header = f"{'Input':>8} {'Output':>8} {'Conc':>6}"
    for mode in available_modes:
        header += f" {mode + '(out)':>16}"
    for mode in available_modes[1:]:
        header += f" {mode + '/'+base_mode:>16}"
    print(header)
    print("-" * 110)

    for key in key_order:
        row = f"{key[0]:>8} {key[1]:>8} {key[2]:>6}"
        mode_rows = {
            mode: next(
                r
                for r in results_by_mode[mode]
                if (r.input_len, r.output_len, r.concurrency) == key
            )
            for mode in available_modes
        }
        for mode in available_modes:
            row += f" {mode_rows[mode].throughput_tokens_per_sec:>16.2f}"
        base = mode_rows[base_mode].throughput_tokens_per_sec
        for mode in available_modes[1:]:
            speedup = mode_rows[mode].throughput_tokens_per_sec / base if base else 0.0
            row += f" {speedup:>16.2f}x"
        print(row)
    print("-" * 110)


def print_vllm_vs_nano(
    vllm_results: List[BenchmarkResult],
    nano_results: List[BenchmarkResult],
    nano_label: str,
) -> None:
    vllm_map = {
        (r.input_len, r.output_len, r.concurrency): r for r in vllm_results
    }
    nano_map = {
        (r.input_len, r.output_len, r.concurrency): r for r in nano_results
    }
    common = sorted(set(vllm_map) & set(nano_map))
    if not common:
        return

    print("\n" + "=" * 110)
    print(f"vLLM vs nano-vLLM ({nano_label})")
    print("-" * 110)
    print(
        f"{'Input':>8} {'Output':>8} {'Conc':>6} "
        f"{'vLLM(out)':>14} {nano_label + '(out)':>18} {'vLLM/nano':>12}"
    )
    print("-" * 110)
    for key in common:
        v = vllm_map[key]
        n = nano_map[key]
        speedup = v.throughput_tokens_per_sec / n.throughput_tokens_per_sec if n.throughput_tokens_per_sec else 0.0
        print(
            f"{key[0]:>8} {key[1]:>8} {key[2]:>6} "
            f"{v.throughput_tokens_per_sec:>14.2f} {n.throughput_tokens_per_sec:>18.2f} {speedup:>12.2f}x"
        )
    print("-" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM and nano-vLLM")
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=["vllm", "nano-vllm", "both"],
        help="Engine to benchmark",
    )
    parser.add_argument(
        "--nano-mode",
        type=str,
        default="all",
        choices=["eager", "batched", "cudagraph", "all"],
        help="nano-vLLM execution mode to benchmark",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/coding/models/Qwen/Qwen3-0___6B",
        help="Path to the model",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.98,
        help="GPU memory utilization for nano-vLLM",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "full"],
        help="Benchmark config set",
    )
    args = parser.parse_args()

    configs = default_configs(args.profile)

    vllm_results: List[BenchmarkResult] = []
    nano_results_by_mode: Dict[str, List[BenchmarkResult]] = {}

    if args.engine in ["vllm", "both"]:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking vLLM with model: {args.model}")
        print(f"Configs: {configs}")
        print(f"{'=' * 70}\n")
        vllm_results = run_vllm_benchmarks(
            model_path=args.model,
            configs=configs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        with open("vllm_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "engine": "vllm",
                    "model": args.model,
                    "configs": configs,
                    "results": [asdict(r) for r in vllm_results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print_summary(vllm_results, "vLLM Summary")

    if args.engine in ["nano-vllm", "both"]:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking nano-vLLM with model: {args.model}")
        print(f"Modes: {['eager', 'batched', 'cudagraph'] if args.nano_mode == 'all' else [args.nano_mode]}")
        print(f"Configs: {configs}")
        print(f"{'=' * 70}\n")
        nano_results_by_mode = run_nano_vllm_benchmarks(
            model_path=args.model,
            configs=configs,
            nano_mode=args.nano_mode,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        with open("nano_vllm_mode_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "engine": "nano-vllm",
                    "model": args.model,
                    "configs": configs,
                    "results_by_mode": {
                        mode: [asdict(r) for r in results]
                        for mode, results in nano_results_by_mode.items()
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        flat_results = [result for results in nano_results_by_mode.values() for result in results]
        print_summary(flat_results, "nano-vLLM Summary")
        print_nano_mode_comparison(nano_results_by_mode)

    if args.engine == "both" and vllm_results:
        if args.nano_mode == "all":
            preferred = "cudagraph" if "cudagraph" in nano_results_by_mode else next(iter(nano_results_by_mode))
            print_vllm_vs_nano(vllm_results, nano_results_by_mode[preferred], preferred)
        else:
            print_vllm_vs_nano(vllm_results, nano_results_by_mode[args.nano_mode], args.nano_mode)


if __name__ == "__main__":
    main()
