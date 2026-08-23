from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import csv
import gc
import hashlib
import json
import math
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
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path.cwd()
AUDIT = ROOT / 'results' / 'v2' / 'audit'
PROTOCOL_VERSION = 'hgb_cpu_deployment_benchmark_v3_4'

HGB_MODEL = ROOT / 'results' / 'v2' / 'tabular_baselines' / 'runs' / 'hist_gradient_boosting_b0__seed_2026' / 'model.joblib'
HGB_VERIFICATION = AUDIT / 'hist_gradient_boosting_b0_verification_v3_2.json'
PHASE6_SUMMARY = AUDIT / 'phase6_deployment_benchmark_summary_v3_2.json'
X_TRAIN = ROOT / 'data' / 'cache' / 'tabular_baseline_family3_v3_2' / 'X_train.npy'

OUTPUT_ROOT = ROOT / 'results' / 'v2' / 'hgb_cpu_deployment_benchmark_v3_4'
RUNS_ROOT = OUTPUT_ROOT / 'runs'
OUTPUT_RUNS = AUDIT / 'hgb_cpu_deployment_benchmark_all_runs_v3_4.csv'
OUTPUT_SUMMARY = AUDIT / 'hgb_cpu_deployment_benchmark_summary_v3_4.json'
OUTPUT_LOCK = AUDIT / 'hgb_cpu_deployment_benchmark_locked_v3_4.json'

REPEATS = (1, 2, 3, 4, 5)
EXPECTED_TRAIN_SHAPE = (1534583, 115)
START = 2026
WARMUP = {1: 50, 32: 30}
MEASURED = {1: 500, 32: 200}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def check_sources() -> None:
    for p in (HGB_MODEL, HGB_VERIFICATION, PHASE6_SUMMARY, X_TRAIN):
        if not p.exists():
            raise FileNotFoundError(p)

    ver = read_json(HGB_VERIFICATION)
    if ver.get('all_checks_passed') is not True:
        raise RuntimeError('Locked HGB verification is not passed.')

    p6 = read_json(PHASE6_SUMMARY)
    policy = p6['runtime_policy']
    checks = {
        'phase6_completed': p6.get('status') == 'completed',
        'cpu_threads_1': int(policy['cpu_threads']) == 1,
        'interop_threads_1': int(policy['interop_threads']) == 1,
        'b1_warmup_50': int(policy['batch_1_warmup_iterations']) == 50,
        'b1_measured_500': int(policy['batch_1_measured_iterations']) == 500,
        'b32_warmup_30': int(policy['batch_32_warmup_iterations']) == 30,
        'b32_measured_200': int(policy['batch_32_measured_iterations']) == 200,
        'validation_false': p6['data_access']['validation_data_access'] is False,
        'test_false': p6['data_access']['test_data_access'] is False,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError('Phase-6 policy mismatch: ' + ', '.join(failed))


def stats_ns(raw: np.ndarray) -> dict[str, float]:
    ms = raw.astype(np.float64) / 1e6
    return {
        'mean_ms': float(np.mean(ms)),
        'median_ms': float(np.median(ms)),
        'std_ms': float(np.std(ms, ddof=0)),
        'p95_ms': float(np.percentile(ms, 95)),
        'p99_ms': float(np.percentile(ms, 99)),
        'min_ms': float(np.min(ms)),
        'max_ms': float(np.max(ms)),
    }


def samples_per_second(raw: np.ndarray, batch: int) -> float:
    sec = float(np.sum(raw, dtype=np.int64)) / 1e9
    return float(batch * len(raw) / sec)


def child(repeat_id: int) -> None:
    check_sources()
    run_dir = RUNS_ROOT / f'repeat_{repeat_id:02d}'
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    x = np.load(X_TRAIN, mmap_mode='r')
    if x.shape != EXPECTED_TRAIN_SHAPE:
        raise RuntimeError(f'Unexpected X_train shape: {x.shape}')
    rows = np.asarray(x[START:START+32], dtype=np.float32).copy()
    if not np.isfinite(rows).all():
        raise RuntimeError('Non-finite benchmark input.')

    proc = psutil.Process(os.getpid())
    gc.collect()
    rss_before = int(proc.memory_info().rss)

    t0 = time.perf_counter_ns()
    with threadpool_limits(limits=1):
        model = joblib.load(HGB_MODEL)
    load_ns = time.perf_counter_ns() - t0

    if not isinstance(model, HistGradientBoostingClassifier):
        raise TypeError(type(model))
    if int(model.n_features_in_) != 115:
        raise RuntimeError('HGB feature count mismatch.')

    rss_after = int(proc.memory_info().rss)
    peak_rss = rss_after
    raw_map: dict[int, np.ndarray] = {}
    result_batches: dict[str, Any] = {}

    with threadpool_limits(limits=1):
        for batch in (1, 32):
            xb = np.ascontiguousarray(rows[:batch])
            for _ in range(WARMUP[batch]):
                out = model.predict_proba(xb)
                if out.shape != (batch, 3) or not np.isfinite(out).all():
                    raise RuntimeError('Invalid HGB warmup output.')
                peak_rss = max(peak_rss, int(proc.memory_info().rss))

            raw = np.empty(MEASURED[batch], dtype=np.int64)
            for i in range(MEASURED[batch]):
                start = time.perf_counter_ns()
                out = model.predict_proba(xb)
                raw[i] = time.perf_counter_ns() - start
                if not np.isfinite(out).all():
                    raise RuntimeError('Invalid HGB measured output.')
                peak_rss = max(peak_rss, int(proc.memory_info().rss))

            if np.any(raw <= 0):
                raise RuntimeError('Non-positive latency.')
            raw_map[batch] = raw
            s = stats_ns(raw)
            result_batches[str(batch)] = {
                'warmup_iterations': WARMUP[batch],
                'measured_iterations': MEASURED[batch],
                **s,
                'samples_per_second': samples_per_second(raw, batch),
            }

    raw_path = run_dir / 'raw_latencies_ns.npz'
    np.savez_compressed(raw_path, batch_1=raw_map[1], batch_32=raw_map[32])

    metrics = {
        'status': 'completed',
        'protocol_version': PROTOCOL_VERSION,
        'repeat_id': repeat_id,
        'model_id': 'hist_gradient_boosting_b0',
        'model_seed': 2026,
        'inference_operation': 'predict_proba',
        'input_space': 'canonical_float32_unscaled',
        'model_artifact': {
            'path': str(HGB_MODEL),
            'sha256': sha256_file(HGB_MODEL),
            'joblib_file_bytes': int(HGB_MODEL.stat().st_size),
        },
        'runtime': {
            'cpu_threads': 1,
            'interop_threads': 1,
            'model_load_elapsed_ms': load_ns / 1e6,
            'model_load_rss_delta_bytes': max(0, rss_after - rss_before),
            'peak_inference_rss_delta_bytes': max(0, peak_rss - rss_after),
        },
        'batch_1': result_batches['1'],
        'batch_32': result_batches['32'],
        'data_access': {
            'train_data_access': True,
            'validation_data_access': False,
            'test_data_access': False,
        },
        'completed_at_utc': utc_now(),
        'all_checks_passed': True,
    }
    atomic_json(run_dir / 'deployment_metrics.json', metrics)
    atomic_json(run_dir / 'run_manifest.json', {
        'status': 'completed',
        'protocol_version': PROTOCOL_VERSION,
        'repeat_id': repeat_id,
        'source_hgb_model_sha256': sha256_file(HGB_MODEL),
        'deployment_metrics_sha256': sha256_file(run_dir / 'deployment_metrics.json'),
        'raw_latencies_sha256': sha256_file(raw_path),
        'validation_data_access': False,
        'test_data_access': False,
        'all_checks_passed': True,
    })

    print(
        f'repeat={repeat_id} | '
        f'batch1={metrics["batch_1"]["median_ms"]:.6f} ms | '
        f'batch32={metrics["batch_32"]["samples_per_second"]:.2f} samples/s | '
        f'joblib={metrics["model_artifact"]["joblib_file_bytes"] / 1024:.3f} KiB',
        flush=True,
    )


def parent(overwrite: bool) -> None:
    check_sources()

    if overwrite:
        for p in (OUTPUT_RUNS, OUTPUT_SUMMARY, OUTPUT_LOCK):
            if p.exists():
                p.unlink()
        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
    else:
        existing = [p for p in (OUTPUT_RUNS, OUTPUT_SUMMARY, OUTPUT_LOCK) if p.exists()]
        if existing or OUTPUT_ROOT.exists():
            raise FileExistsError('Existing v3.4 HGB benchmark outputs found. Review before using --overwrite.')

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    print('=' * 92)
    print('HGB CPU DEPLOYMENT BENCHMARK v3.4')
    print('=' * 92)
    print('Scientific role               : post-selection descriptive comparison')
    print('Locked HGB model seed         : 2026')
    print('Independent process repeats   : 5')
    print('CPU / interop threads         : 1 / 1')
    print('Batch-1 warmup / measured     : 50 / 500')
    print('Batch-32 warmup / measured    : 30 / 200')
    print('Inference operation           : predict_proba')
    print('Validation / test access      : False / False')
    print()

    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'

    for repeat_id in REPEATS:
        print(f'[{repeat_id}/5] starting isolated process...', flush=True)
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--child', '--repeat-id', str(repeat_id)],
            cwd=str(ROOT), env=env, check=False,
        ).returncode
        if rc != 0:
            raise RuntimeError(f'Child repeat {repeat_id} failed, exit={rc}')

    rows: list[dict[str, Any]] = []
    for repeat_id in REPEATS:
        run_dir = RUNS_ROOT / f'repeat_{repeat_id:02d}'
        metrics_path = run_dir / 'deployment_metrics.json'
        raw_path = run_dir / 'raw_latencies_ns.npz'
        manifest_path = run_dir / 'run_manifest.json'
        for p in (metrics_path, raw_path, manifest_path):
            if not p.exists():
                raise FileNotFoundError(p)

        m = read_json(metrics_path)
        with np.load(raw_path) as z:
            b1 = np.array(z['batch_1'], copy=True)
            b32 = np.array(z['batch_32'], copy=True)
        if b1.shape != (500,) or b32.shape != (200,):
            raise RuntimeError('Raw latency shape mismatch.')
        s1, s32 = stats_ns(b1), stats_ns(b32)

        rows.append({
            'repeat_id': repeat_id,
            'model_seed': 2026,
            'joblib_file_bytes': int(m['model_artifact']['joblib_file_bytes']),
            'model_load_elapsed_ms': float(m['runtime']['model_load_elapsed_ms']),
            'model_load_rss_delta_bytes': int(m['runtime']['model_load_rss_delta_bytes']),
            'peak_inference_rss_delta_bytes': int(m['runtime']['peak_inference_rss_delta_bytes']),
            'batch_1_median_ms': s1['median_ms'],
            'batch_1_p95_ms': s1['p95_ms'],
            'batch_1_p99_ms': s1['p99_ms'],
            'batch_1_samples_per_second': samples_per_second(b1, 1),
            'batch_32_median_ms': s32['median_ms'],
            'batch_32_p95_ms': s32['p95_ms'],
            'batch_32_p99_ms': s32['p99_ms'],
            'batch_32_samples_per_second': samples_per_second(b32, 32),
            'validation_data_access': False,
            'test_data_access': False,
            'deployment_metrics_sha256': sha256_file(metrics_path),
            'raw_latencies_sha256': sha256_file(raw_path),
            'run_manifest_sha256': sha256_file(manifest_path),
            'all_checks_passed': True,
        })

    atomic_csv(OUTPUT_RUNS, rows)

    def agg(field: str) -> dict[str, float]:
        arr = np.asarray([float(r[field]) for r in rows], dtype=np.float64)
        return {
            'mean': float(np.mean(arr)),
            'std_population': float(np.std(arr, ddof=0)),
            'minimum': float(np.min(arr)),
            'maximum': float(np.max(arr)),
        }

    summary = {
        'status': 'completed',
        'protocol_version': PROTOCOL_VERSION,
        'completed_at_utc': utc_now(),
        'scientific_role': 'post-selection descriptive CPU deployment comparison; does not alter TinyML selection',
        'model_id': 'hist_gradient_boosting_b0',
        'model_seed': 2026,
        'benchmark_repeat_count': 5,
        'repeat_unit': 'independent process benchmark repeats of the same locked model',
        'joblib_file_bytes': int(rows[0]['joblib_file_bytes']),
        'metrics': {
            field: agg(field) for field in (
                'model_load_elapsed_ms',
                'model_load_rss_delta_bytes',
                'peak_inference_rss_delta_bytes',
                'batch_1_median_ms',
                'batch_1_p95_ms',
                'batch_1_p99_ms',
                'batch_1_samples_per_second',
                'batch_32_median_ms',
                'batch_32_p95_ms',
                'batch_32_p99_ms',
                'batch_32_samples_per_second',
            )
        },
        'runtime_policy': {
            'cpu_threads': 1,
            'interop_threads': 1,
            'batch_1_warmup_iterations': 50,
            'batch_1_measured_iterations': 500,
            'batch_32_warmup_iterations': 30,
            'batch_32_measured_iterations': 200,
            'inference_operation': 'predict_proba',
        },
        'data_access': {
            'train_data_access': True,
            'validation_data_access': False,
            'test_data_access': False,
        },
        'comparability_note': (
            'HGB uses its locked canonical unscaled float32 input. Phase-6 TinyML uses train-only standardized float32 input. '
            'HGB uncertainty here is five process repeats of one trained model, whereas Phase-6 TinyML group summaries span five matched training seeds.'
        ),
        'all_runs_csv': {
            'path': str(OUTPUT_RUNS),
            'sha256': sha256_file(OUTPUT_RUNS),
        },
        'all_checks_passed': True,
    }
    atomic_json(OUTPUT_SUMMARY, summary)

    lock = {
        'status': 'locked',
        'protocol_version': PROTOCOL_VERSION,
        'locked_at_utc': utc_now(),
        'tinyml_model_family_reselected': False,
        'hgb_benchmark_repeat_count': 5,
        'validation_data_access': False,
        'test_data_access': False,
        'summary_path': str(OUTPUT_SUMMARY),
        'summary_sha256': sha256_file(OUTPUT_SUMMARY),
        'all_runs_path': str(OUTPUT_RUNS),
        'all_runs_sha256': sha256_file(OUTPUT_RUNS),
        'all_checks_passed': True,
    }
    atomic_json(OUTPUT_LOCK, lock)

    print()
    print('=' * 92)
    print('HGB CPU DEPLOYMENT BENCHMARK SUMMARY')
    print('=' * 92)
    print(f'Joblib artifact                 : {summary["joblib_file_bytes"] / 1024:.3f} KiB')
    print(
        'Batch-1 median mean ± std       : '
        f'{summary["metrics"]["batch_1_median_ms"]["mean"]:.6f} ± '
        f'{summary["metrics"]["batch_1_median_ms"]["std_population"]:.6f} ms'
    )
    print(
        'Batch-1 p95 mean                : '
        f'{summary["metrics"]["batch_1_p95_ms"]["mean"]:.6f} ms'
    )
    print(
        'Batch-32 throughput mean        : '
        f'{summary["metrics"]["batch_32_samples_per_second"]["mean"]:.3f} samples/s'
    )
    print(
        'Peak inference RSS Δ mean       : '
        f'{summary["metrics"]["peak_inference_rss_delta_bytes"]["mean"] / (1024**2):.3f} MiB'
    )
    print('Validation / test access        : False / False')
    print('TinyML model-family reselected  : False')
    print('All checks passed               : True')
    print('Summary                         :', OUTPUT_SUMMARY)
    print('Lock                            :', OUTPUT_LOCK)
    print('HGB CPU DEPLOYMENT BENCHMARK COMPLETED')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--repeat-id', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.child:
        if args.repeat_id not in REPEATS:
            raise ValueError('--repeat-id must be 1..5')
        child(args.repeat_id)
    else:
        parent(args.overwrite)


if __name__ == '__main__':
    main()
