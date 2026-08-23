from __future__ import annotations

import os

# Enforce single-thread execution before numerical libraries are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_VERSION = "hgb_p50_fair_decision_benchmark_v3_5"

PHASE6_SCRIPT = (
    ROOT / "scripts" / "152_run_phase6_isolated_deployment_benchmark_v3_2.py"
)

HGB_MODEL = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
    / "hist_gradient_boosting_b0__seed_2026"
    / "model.joblib"
)

OUTPUT_ROOT = ROOT / "results" / "v2" / "hgb_p50_fair_decision_benchmark_v3_5"
RUNS_ROOT = OUTPUT_ROOT / "runs"

OUTPUT_ALL_RUNS = AUDIT / "hgb_p50_fair_decision_benchmark_all_runs_v3_5.csv"
OUTPUT_SUMMARY = AUDIT / "hgb_p50_fair_decision_benchmark_summary_v3_5.json"
OUTPUT_LOCK = AUDIT / "hgb_p50_fair_decision_benchmark_locked_v3_5.json"
OUTPUT_MANIFEST = AUDIT / "hgb_p50_fair_decision_benchmark_manifest_v3_5.json"

MODEL_KEYS = ("tinyml_mlp_P50_QAT", "hist_gradient_boosting_B0")
REPEAT_IDS = (1, 2, 3, 4, 5)

BATCH_SIZES = (1, 32)
WARMUP_ITERATIONS = {1: 50, 32: 30}
MEASURED_ITERATIONS = {1: 500, 32: 200}

CPU_THREADS = 1
INTEROP_THREADS = 1
EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_TRAIN_ROWS = 1_534_583
TRAIN_INPUT_START_INDEX = 2026


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No CSV rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_phase6_module():
    if not PHASE6_SCRIPT.exists():
        raise FileNotFoundError(PHASE6_SCRIPT)
    spec = importlib.util.spec_from_file_location(
        "phase6_reference_benchmark_v3_2",
        PHASE6_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import Phase-6 reference benchmark.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std_population": float(np.std(arr, ddof=0)),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
    }


def latency_stats(ns: np.ndarray) -> dict[str, float]:
    ms = ns.astype(np.float64) / 1_000_000.0
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(np.mean(ms)),
    }


def throughput(ns: np.ndarray, batch_size: int) -> float:
    seconds = float(np.sum(ns, dtype=np.int64)) / 1e9
    return float(batch_size * len(ns) / seconds)


def run_dir(model_key: str, repeat_id: int) -> Path:
    return RUNS_ROOT / model_key / f"repeat_{repeat_id:02d}"


def prepare_shared_inputs(phase6):
    x_train = np.load(phase6.X_TRAIN_PATH, mmap_mode="r")
    if x_train.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURE_COUNT):
        raise RuntimeError(f"Unexpected train shape: {x_train.shape}")

    raw = np.asarray(
        x_train[
            TRAIN_INPUT_START_INDEX:
            TRAIN_INPUT_START_INDEX + 32
        ],
        dtype=np.float32,
    ).copy()

    with np.load(phase6.SCALER_NPZ) as values:
        mean64 = np.array(values["mean_float64"], copy=True)
        scale64 = np.array(values["scale_float64"], copy=True)

    scaled = phase6.transform_rows(raw, mean64, scale64)

    return raw, scaled


def load_p50_model(phase6, scaled: np.ndarray):
    rows = phase6.read_csv(phase6.PHASE5_MASTER_MATRIX)
    matched = [
        row for row in rows
        if row["architecture"] == "tinyml_mlp"
        and row["variant"] == "P50-QAT"
        and int(row["seed"]) == 2026
    ]
    if len(matched) != 1:
        raise RuntimeError(
            f"Expected one TinyML P50-QAT seed-2026 row; found={len(matched)}"
        )
    row = matched[0]

    metrics_path = Path(row["metrics_path"])
    metrics = phase6.read_json(metrics_path)

    b0_rows = phase6.read_csv(phase6.B0_REGISTRY)
    pruning_rows = phase6.read_csv(phase6.PRUNING_REGISTRY)

    b0_lookup = {
        (r["architecture"], int(r["seed"])): r
        for r in b0_rows
    }
    pruning_lookup = {
        (r["architecture"], r["variant"], int(r["seed"])): r
        for r in pruning_rows
    }

    unique_suffix = f"fair_{os.getpid()}"

    model_module = phase6.load_module(
        phase6.MODEL_SOURCE,
        f"fair_models_{unique_suffix}",
    )
    pruning_engine_module = phase6.load_module(
        phase6.PRUNING_ENGINE_PATH,
        f"fair_pruning_{unique_suffix}",
    )
    qat_engine_module = phase6.load_module(
        phase6.QAT_ENGINE_PATH,
        f"fair_qat_{unique_suffix}",
    )
    ptq_engine_module = phase6.load_module(
        phase6.PTQ_ENGINE_PATH,
        f"fair_ptq_{unique_suffix}",
    )

    structure_tensor = torch.from_numpy(scaled[:32])

    model, checkpoint_path, checkpoint_hash, representation = (
        phase6.load_deployment_model(
            "tinyml_mlp",
            "P50-QAT",
            2026,
            Path(row["run_directory"]),
            metrics,
            model_module,
            pruning_engine_module,
            qat_engine_module,
            ptq_engine_module,
            b0_lookup,
            pruning_lookup,
            structure_tensor,
        )
    )

    if representation != "static_int8":
        raise RuntimeError(f"Unexpected P50-QAT representation: {representation}")

    return model, checkpoint_path, checkpoint_hash


def benchmark_child(model_key: str, repeat_id: int) -> None:
    if model_key not in MODEL_KEYS:
        raise ValueError(model_key)
    if repeat_id not in REPEAT_IDS:
        raise ValueError(repeat_id)

    phase6 = load_phase6_module()

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(INTEROP_THREADS)
    except RuntimeError:
        pass

    raw, scaled = prepare_shared_inputs(phase6)

    out_dir = run_dir(model_key, repeat_id)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    process = psutil.Process(os.getpid())
    gc.collect()

    if model_key == "tinyml_mlp_P50_QAT":
        model, artifact_path, artifact_hash = load_p50_model(phase6, scaled)
        model.eval()

        inputs = {
            1: torch.from_numpy(scaled[:1]),
            32: torch.from_numpy(scaled[:32]),
        }

        def infer(batch_size: int):
            with torch.inference_mode():
                logits = phase6.extract_logits(model(inputs[batch_size]))
                labels = torch.argmax(logits, dim=1)
            return labels.detach().cpu().numpy()

        inference_operation = "argmax(model_logits)"
        input_contract = "train-only standardized float32"
        native_artifact_bytes = int(
            phase6.serialized_state_size(model)
        )
        native_artifact_kind = "PyTorch serialized state_dict"

    else:
        if not HGB_MODEL.exists():
            raise FileNotFoundError(HGB_MODEL)

        with threadpool_limits(limits=1):
            model = joblib.load(HGB_MODEL)

        inputs = {
            1: np.ascontiguousarray(raw[:1]),
            32: np.ascontiguousarray(raw[:32]),
        }

        def infer(batch_size: int):
            with threadpool_limits(limits=1):
                return np.asarray(model.predict(inputs[batch_size]))

        inference_operation = "predict"
        input_contract = "canonical unscaled float32"
        artifact_path = HGB_MODEL
        artifact_hash = sha256_file(HGB_MODEL)
        native_artifact_bytes = int(HGB_MODEL.stat().st_size)
        native_artifact_kind = "compressed joblib file"

    latency_arrays = {}
    batch_results = {}
    output_hashes = {}

    peak_rss = int(process.memory_info().rss)
    rss_before_benchmark = peak_rss

    for batch_size in BATCH_SIZES:
        for _ in range(WARMUP_ITERATIONS[batch_size]):
            labels = infer(batch_size)
            if labels.shape != (batch_size,):
                raise RuntimeError(
                    f"{model_key} warm-up output shape: {labels.shape}"
                )

        latencies = np.empty(
            MEASURED_ITERATIONS[batch_size],
            dtype=np.int64,
        )
        final_labels = None

        for i in range(MEASURED_ITERATIONS[batch_size]):
            started = time.perf_counter_ns()
            final_labels = infer(batch_size)
            latencies[i] = time.perf_counter_ns() - started
            peak_rss = max(peak_rss, int(process.memory_info().rss))

        if final_labels is None:
            raise RuntimeError("No measured output.")
        if final_labels.shape != (batch_size,):
            raise RuntimeError(
                f"{model_key} output shape: {final_labels.shape}"
            )
        if not np.isin(final_labels, [0, 1, 2]).all():
            raise RuntimeError(f"Unexpected predicted class in {model_key}.")

        stats = latency_stats(latencies)
        batch_results[batch_size] = {
            **stats,
            "samples_per_second": throughput(latencies, batch_size),
        }
        latency_arrays[batch_size] = latencies
        output_hashes[batch_size] = hashlib.sha256(
            np.asarray(final_labels, dtype=np.int64).tobytes()
        ).hexdigest()

    raw_latency_path = out_dir / "raw_latencies_ns.npz"
    np.savez_compressed(
        raw_latency_path,
        batch_1=latency_arrays[1],
        batch_32=latency_arrays[32],
    )

    metrics = {
        "status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "repeat_id": repeat_id,
        "model_seed": 2026,
        "scientific_role": (
            "post-selection fairness benchmark; no model reselection"
        ),
        "inference_operation": inference_operation,
        "output_semantics": "predicted class labels",
        "input_contract": input_contract,
        "input_source": str(phase6.X_TRAIN_PATH),
        "input_source_row_start": TRAIN_INPUT_START_INDEX,
        "input_source_row_count": 32,
        "preprocessing_inside_timed_region": False,
        "model_loading_inside_timed_region": False,
        "batch_1": {
            "warmup_iterations": WARMUP_ITERATIONS[1],
            "measured_iterations": MEASURED_ITERATIONS[1],
            **batch_results[1],
            "output_sha256": output_hashes[1],
        },
        "batch_32": {
            "warmup_iterations": WARMUP_ITERATIONS[32],
            "measured_iterations": MEASURED_ITERATIONS[32],
            **batch_results[32],
            "output_sha256": output_hashes[32],
        },
        "native_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "bytes": native_artifact_bytes,
            "kind": native_artifact_kind,
            "format_neutral_comparison": False,
        },
        "runtime": {
            "cpu_threads": CPU_THREADS,
            "interop_threads": INTEROP_THREADS,
            "rss_before_benchmark_bytes": rss_before_benchmark,
            "peak_process_rss_bytes": peak_rss,
            "peak_benchmark_rss_delta_bytes": max(
                0, peak_rss - rss_before_benchmark
            ),
        },
        "data_access": {
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "completed_at_utc": utc_now(),
        "all_checks_passed": True,
    }

    metrics_path = out_dir / "deployment_metrics.json"
    atomic_json(metrics_path, metrics)

    print(
        f"{model_key} repeat={repeat_id} | "
        f"batch1={batch_results[1]['median_ms']:.6f} ms | "
        f"batch32={batch_results[32]['samples_per_second']:.2f} samples/s",
        flush=True,
    )


def parent(overwrite: bool) -> None:
    required = [PHASE6_SCRIPT, HGB_MODEL]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(map(str, missing))
        )

    aggregate_paths = (
        OUTPUT_ALL_RUNS,
        OUTPUT_SUMMARY,
        OUTPUT_LOCK,
        OUTPUT_MANIFEST,
    )

    if overwrite:
        for path in aggregate_paths:
            if path.exists():
                path.unlink()
        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
    else:
        existing = [p for p in aggregate_paths if p.exists()]
        existing += [
            run_dir(model_key, repeat_id)
            for model_key in MODEL_KEYS
            for repeat_id in REPEAT_IDS
            if run_dir(model_key, repeat_id).exists()
        ]
        if existing:
            raise FileExistsError(
                "Existing v3.5 outputs found. Review before --overwrite:\n"
                + "\n".join(map(str, existing))
            )

    print("=" * 96)
    print("HGB vs P50-QAT FAIR CLASS-DECISION CPU BENCHMARK v3.5")
    print("=" * 96)
    print("Models                           : P50-QAT seed 2026 vs HGB seed 2026")
    print("Repeat unit                      : 5 isolated process repeats/model")
    print("Output semantics                 : predicted class labels")
    print("TinyML timed operation           : argmax(model logits)")
    print("HGB timed operation              : predict")
    print("Preprocessing in timed region    : False")
    print("Model loading in timed region    : False")
    print("CPU / interop threads            : 1 / 1")
    print("Validation / test access         : False / False")
    print("TinyML family reselected         : False")
    print()

    for model_key in MODEL_KEYS:
        for repeat_id in REPEAT_IDS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--model-key",
                model_key,
                "--repeat-id",
                str(repeat_id),
            ]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"

            print(
                f"[{model_key} {repeat_id}/5] starting isolated process...",
                flush=True,
            )
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Child failed: {model_key} repeat={repeat_id}, "
                    f"exit={completed.returncode}"
                )

    rows = []
    for model_key in MODEL_KEYS:
        for repeat_id in REPEAT_IDS:
            path = run_dir(model_key, repeat_id) / "deployment_metrics.json"
            metrics = read_json(path)
            if metrics.get("all_checks_passed") is not True:
                raise RuntimeError(f"Failed child checks: {path}")
            if metrics["data_access"]["validation_data_access"] is not False:
                raise RuntimeError("Validation data accessed.")
            if metrics["data_access"]["test_data_access"] is not False:
                raise RuntimeError("Test data accessed.")

            rows.append(
                {
                    "model_key": model_key,
                    "repeat_id": repeat_id,
                    "model_seed": 2026,
                    "inference_operation": metrics["inference_operation"],
                    "output_semantics": metrics["output_semantics"],
                    "batch_1_median_ms": float(
                        metrics["batch_1"]["median_ms"]
                    ),
                    "batch_1_p95_ms": float(
                        metrics["batch_1"]["p95_ms"]
                    ),
                    "batch_32_samples_per_second": float(
                        metrics["batch_32"]["samples_per_second"]
                    ),
                    "native_artifact_bytes": int(
                        metrics["native_artifact"]["bytes"]
                    ),
                    "native_artifact_kind": metrics["native_artifact"]["kind"],
                    "validation_data_access": False,
                    "test_data_access": False,
                    "metrics_sha256": sha256_file(path),
                    "all_checks_passed": True,
                }
            )

    atomic_csv(OUTPUT_ALL_RUNS, rows)

    by_model = {}
    for model_key in MODEL_KEYS:
        subset = [r for r in rows if r["model_key"] == model_key]
        by_model[model_key] = {
            "repeat_count": len(subset),
            "batch_1_median_ms": aggregate(
                [float(r["batch_1_median_ms"]) for r in subset]
            ),
            "batch_1_p95_ms": aggregate(
                [float(r["batch_1_p95_ms"]) for r in subset]
            ),
            "batch_32_samples_per_second": aggregate(
                [float(r["batch_32_samples_per_second"]) for r in subset]
            ),
            "native_artifact_bytes": int(subset[0]["native_artifact_bytes"]),
            "native_artifact_kind": subset[0]["native_artifact_kind"],
        }

    p50 = by_model["tinyml_mlp_P50_QAT"]
    hgb = by_model["hist_gradient_boosting_B0"]

    ratios = {
        "hgb_over_p50_batch1_latency": (
            hgb["batch_1_median_ms"]["mean"]
            / p50["batch_1_median_ms"]["mean"]
        ),
        "p50_over_hgb_batch32_throughput": (
            p50["batch_32_samples_per_second"]["mean"]
            / hgb["batch_32_samples_per_second"]["mean"]
        ),
    }

    summary = {
        "status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "scientific_role": (
            "post-selection operation-matched CPU comparison"
        ),
        "comparison_definition": {
            "same_model_seed": 2026,
            "same_repeat_unit": "five isolated process repeats per locked model",
            "same_output_semantics": "predicted class labels",
            "preprocessing_inside_timed_region": False,
            "model_loading_inside_timed_region": False,
            "native_storage_format_neutral": False,
        },
        "models": by_model,
        "descriptive_ratios": ratios,
        "data_access": {
            "validation_data_access": False,
            "test_data_access": False,
        },
        "tinyml_model_family_reselected": False,
        "all_runs_csv": {
            "path": str(OUTPUT_ALL_RUNS),
            "sha256": sha256_file(OUTPUT_ALL_RUNS),
        },
        "completed_at_utc": utc_now(),
        "all_checks_passed": True,
    }
    atomic_json(OUTPUT_SUMMARY, summary)

    lock = {
        "status": "locked",
        "protocol_version": PROTOCOL_VERSION,
        "summary_path": str(OUTPUT_SUMMARY),
        "summary_sha256": sha256_file(OUTPUT_SUMMARY),
        "all_runs_path": str(OUTPUT_ALL_RUNS),
        "all_runs_sha256": sha256_file(OUTPUT_ALL_RUNS),
        "tinyml_model_family_reselected": False,
        "validation_data_access": False,
        "test_data_access": False,
        "all_checks_passed": True,
        "locked_at_utc": utc_now(),
    }
    atomic_json(OUTPUT_LOCK, lock)

    manifest = {
        "status": "locked",
        "protocol_version": PROTOCOL_VERSION,
        "source_phase6_script": {
            "path": str(PHASE6_SCRIPT),
            "sha256": sha256_file(PHASE6_SCRIPT),
        },
        "source_hgb_model": {
            "path": str(HGB_MODEL),
            "sha256": sha256_file(HGB_MODEL),
        },
        "generated": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in (
                OUTPUT_ALL_RUNS,
                OUTPUT_SUMMARY,
                OUTPUT_LOCK,
            )
        ],
        "all_checks_passed": True,
        "created_at_utc": utc_now(),
    }
    atomic_json(OUTPUT_MANIFEST, manifest)

    print()
    print("=" * 96)
    print("FAIR CLASS-DECISION BENCHMARK SUMMARY")
    print("=" * 96)
    print(
        "P50-QAT batch-1 median mean ± std : "
        f"{p50['batch_1_median_ms']['mean']:.6f} ± "
        f"{p50['batch_1_median_ms']['std_population']:.6f} ms"
    )
    print(
        "HGB batch-1 median mean ± std     : "
        f"{hgb['batch_1_median_ms']['mean']:.6f} ± "
        f"{hgb['batch_1_median_ms']['std_population']:.6f} ms"
    )
    print(
        "P50-QAT batch-32 throughput mean  : "
        f"{p50['batch_32_samples_per_second']['mean']:.3f} samples/s"
    )
    print(
        "HGB batch-32 throughput mean      : "
        f"{hgb['batch_32_samples_per_second']['mean']:.3f} samples/s"
    )
    print(
        "HGB/P50 batch-1 latency ratio     : "
        f"{ratios['hgb_over_p50_batch1_latency']:.3f}x"
    )
    print(
        "P50/HGB batch-32 throughput ratio : "
        f"{ratios['p50_over_hgb_batch32_throughput']:.3f}x"
    )
    print("Native storage formats comparable : False")
    print("Validation / test access          : False / False")
    print("TinyML family reselected          : False")
    print("All checks passed                 : True")
    print("Summary                           :", OUTPUT_SUMMARY)
    print("Lock                              :", OUTPUT_LOCK)
    print("FAIR HGB vs P50-QAT BENCHMARK COMPLETED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument("--repeat-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.child:
        if args.model_key is None or args.repeat_id is None:
            raise ValueError("--model-key and --repeat-id are required.")
        benchmark_child(args.model_key, args.repeat_id)
    else:
        parent(args.overwrite)


if __name__ == "__main__":
    main()
