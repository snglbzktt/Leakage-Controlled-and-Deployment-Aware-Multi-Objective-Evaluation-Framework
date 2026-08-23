from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path.cwd()
BUILD = ROOT / 'data' / 'cache' / 'tabular_baseline_family3_v3_1.building'
AUDIT_DB = BUILD / 'scaled_model_input_overlap_audit.sqlite'
FAILURE_JSON = BUILD / 'scaled_overlap_failure.json'
OUT_DIR = ROOT / 'results' / 'v2' / 'audit'
OUT_JSON = OUT_DIR / 'tabular_baseline_scaled_repair_plan_v3.json'
OUT_COMPONENTS = OUT_DIR / 'tabular_baseline_scaled_repair_components_v3.csv'
OUT_MOVES = OUT_DIR / 'tabular_baseline_scaled_repair_moves_v3.csv'

FEATURE_COUNT = 115
CHUNK_SIZE = 100_000
SPLITS = ('train', 'validation', 'test')
EXPECTED_TOTAL_FP = 2_278_176
EXPECTED_TOTAL_RAW = 7_062_606
EXPECTED_COMPONENTS = 280
EXPECTED = {
    'train': (1_534_583, 4_943_823),
    'validation': (371_796, 1_059_389),
    'test': (371_797, 1_059_394),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def float32_fingerprints(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(values, dtype='<f4', order='C', copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise RuntimeError(f'Invalid matrix shape: {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise RuntimeError('Non-finite scaled value found.')
    matrix[matrix == 0.0] = 0.0
    words = matrix.view('<u4').reshape(matrix.shape)
    forward = np.full(matrix.shape[0], np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    reverse = np.full(matrix.shape[0], np.uint64(0x84222325CBF29CE4), dtype=np.uint64)
    pf = np.uint64(0x100000001B3)
    pr = np.uint64(0x9E3779B185EBCA87)
    with np.errstate(over='ignore'):
        for column in range(matrix.shape[1]):
            wf = words[:, column].astype(np.uint64, copy=False)
            wr = words[:, matrix.shape[1] - 1 - column].astype(np.uint64, copy=False)
            forward = (forward ^ (wf + np.uint64(column + 1))) * pf
            reverse = (reverse ^ (wr + np.uint64(column + 1))) * pr
            forward ^= forward >> np.uint64(32)
            reverse ^= reverse >> np.uint64(29)
    return forward.view(np.int64), reverse.view(np.int64)


def main() -> None:
    started = time.perf_counter()
    required = [BUILD, AUDIT_DB, FAILURE_JSON, BUILD / 'scaler_mean.npy', BUILD / 'scaler_scale.npy']
    for split in SPLITS:
        required += [
            BUILD / f'X_{split}.npy', BUILD / f'y_{split}.npy',
            BUILD / f'raw_row_count_{split}.npy',
            BUILD / f'hash_forward_{split}.npy', BUILD / f'hash_reverse_{split}.npy',
        ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    failure = json.loads(FAILURE_JSON.read_text(encoding='utf-8'))
    uri = AUDIT_DB.resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True, timeout=120.0) as con:
        quick = str(con.execute('PRAGMA quick_check').fetchone()[0])
        if quick.lower() != 'ok':
            raise RuntimeError(f'Audit DB quick_check failed: {quick}')
        targets = {
            (int(a), int(b))
            for a, b in con.execute('''
                SELECT hash_forward, hash_reverse
                FROM scaled_fingerprints
                GROUP BY hash_forward, hash_reverse
                HAVING COUNT(DISTINCT split_code) > 1
            ''').fetchall()
        }
    if len(targets) != EXPECTED_COMPONENTS:
        raise RuntimeError(f'Unexpected component count: {len(targets)}')

    mean = np.load(BUILD / 'scaler_mean.npy')
    scale = np.load(BUILD / 'scaler_scale.npy')
    components: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    processed = 0

    print('=' * 92)
    print('SCALED COLLISION MINIMUM-MOVEMENT PLAN')
    print('=' * 92)
    print(f'Target components: {len(targets):,}')

    for split_code, split in enumerate(SPLITS):
        x = np.load(BUILD / f'X_{split}.npy', mmap_mode='r')
        y = np.load(BUILD / f'y_{split}.npy', mmap_mode='r')
        raw = np.load(BUILD / f'raw_row_count_{split}.npy', mmap_mode='r')
        ohf = np.load(BUILD / f'hash_forward_{split}.npy', mmap_mode='r')
        ohr = np.load(BUILD / f'hash_reverse_{split}.npy', mmap_mode='r')
        for start in range(0, len(x), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(x))
            scaled = ((np.asarray(x[start:end], dtype=np.float64) - mean) / scale).astype(np.float32)
            hf, hr = float32_fingerprints(scaled)
            for local, key in enumerate(zip(hf.tolist(), hr.tolist())):
                key = (int(key[0]), int(key[1]))
                if key not in targets:
                    continue
                idx = start + local
                components[key].append({
                    'source_split_code': split_code,
                    'source_split': split,
                    'source_row_index': int(idx),
                    'family': int(y[idx]),
                    'raw_row_count': int(raw[idx]),
                    'original_hash_forward': int(ohf[idx]),
                    'original_hash_reverse': int(ohr[idx]),
                })
            processed += end - start
        print(f'[scan] {split:<10} rows={len(x):,}', flush=True)

    if processed != EXPECTED_TOTAL_FP or set(components) != targets:
        raise RuntimeError('Scaled component recovery failed.')

    fp_delta = np.zeros(3, dtype=np.int64)
    raw_delta = np.zeros(3, dtype=np.int64)
    component_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    min_moves = 0
    min_raw_moves = 0

    for number, key in enumerate(sorted(components), start=1):
        rows = components[key]
        families = {int(row['family']) for row in rows}
        if len(families) != 1:
            raise RuntimeError(f'Cross-family collision: {key} -> {families}')
        counts = np.zeros(3, dtype=np.int64)
        raws = np.zeros(3, dtype=np.int64)
        for row in rows:
            s = int(row['source_split_code'])
            counts[s] += 1
            raws[s] += int(row['raw_row_count'])
        if np.count_nonzero(counts) < 2:
            raise RuntimeError(f'Component does not span splits: {key}')

        max_count = int(counts.max())
        candidates = [s for s in range(3) if int(counts[s]) == max_count]
        max_raw = max(int(raws[s]) for s in candidates)
        candidates = [s for s in candidates if int(raws[s]) == max_raw]
        min_moves += int(counts.sum()) - max_count
        min_raw_moves += int(raws.sum()) - max_raw

        scored = []
        for dest in candidates:
            nfp = fp_delta.copy()
            nraw = raw_delta.copy()
            for src in range(3):
                if src == dest:
                    continue
                nfp[src] -= counts[src]
                nfp[dest] += counts[src]
                nraw[src] -= raws[src]
                nraw[dest] += raws[src]
            score = (int(np.abs(nraw).sum()), int(np.abs(nfp).sum()), int(dest))
            scored.append((score, dest, nfp, nraw))
        _, dest, fp_delta, raw_delta = min(scored, key=lambda item: item[0])

        moved_count = int(counts.sum() - counts[dest])
        moved_raw = int(raws.sum() - raws[dest])
        component_rows.append({
            'component_number': number,
            'scaled_hash_forward': key[0],
            'scaled_hash_reverse': key[1],
            'family': next(iter(families)),
            'split_span': int(np.count_nonzero(counts)),
            'source_fingerprint_count': int(counts.sum()),
            'source_raw_row_count': int(raws.sum()),
            'train_fingerprint_count': int(counts[0]),
            'validation_fingerprint_count': int(counts[1]),
            'test_fingerprint_count': int(counts[2]),
            'train_raw_row_count': int(raws[0]),
            'validation_raw_row_count': int(raws[1]),
            'test_raw_row_count': int(raws[2]),
            'destination_split_code': int(dest),
            'destination_split': SPLITS[dest],
            'moved_fingerprint_count': moved_count,
            'moved_raw_row_count': moved_raw,
        })
        for row in rows:
            src = int(row['source_split_code'])
            if src == dest:
                continue
            move_rows.append({
                'component_number': number,
                'scaled_hash_forward': key[0],
                'scaled_hash_reverse': key[1],
                'family': int(row['family']),
                'source_split_code': src,
                'source_split': SPLITS[src],
                'destination_split_code': int(dest),
                'destination_split': SPLITS[dest],
                'source_row_index': int(row['source_row_index']),
                'raw_row_count': int(row['raw_row_count']),
                'original_hash_forward': int(row['original_hash_forward']),
                'original_hash_reverse': int(row['original_hash_reverse']),
            })

    moved_fp = len(move_rows)
    moved_raw = sum(int(row['raw_row_count']) for row in move_rows)
    if moved_fp != min_moves or moved_raw != min_raw_moves:
        raise RuntimeError('Plan is not lexicographically minimal.')

    original_fp = np.array([EXPECTED[s][0] for s in SPLITS], dtype=np.int64)
    original_raw = np.array([EXPECTED[s][1] for s in SPLITS], dtype=np.int64)
    proposed_fp = original_fp + fp_delta
    proposed_raw = original_raw + raw_delta
    if int(proposed_fp.sum()) != EXPECTED_TOTAL_FP or int(proposed_raw.sum()) != EXPECTED_TOTAL_RAW:
        raise RuntimeError('Totals are not preserved.')

    component_fields = list(component_rows[0].keys())
    move_fields = list(move_rows[0].keys())
    atomic_csv(OUT_COMPONENTS, component_rows, component_fields)
    atomic_csv(OUT_MOVES, move_rows, move_fields)

    split_plan = []
    for code, split in enumerate(SPLITS):
        split_plan.append({
            'split_code': code,
            'split': split,
            'original_fingerprint_count': int(original_fp[code]),
            'fingerprint_delta': int(fp_delta[code]),
            'proposed_fingerprint_count': int(proposed_fp[code]),
            'original_raw_row_count': int(original_raw[code]),
            'raw_row_delta': int(raw_delta[code]),
            'proposed_raw_row_count': int(proposed_raw[code]),
        })

    summary = {
        'status': 'plan_only_not_applied',
        'generated_at_utc': utc_now(),
        'cross_split_scaled_component_count': len(component_rows),
        'affected_original_fingerprint_row_count': sum(len(v) for v in components.values()),
        'minimum_moved_fingerprint_count': moved_fp,
        'minimum_moved_raw_row_count_given_minimum_fingerprints': moved_raw,
        'split_plan': split_plan,
        'component_span_distribution': {
            '2': sum(1 for row in component_rows if row['split_span'] == 2),
            '3': sum(1 for row in component_rows if row['split_span'] == 3),
        },
        'family_distribution': {
            str(f): sum(1 for row in component_rows if row['family'] == f)
            for f in range(3)
        },
        'components_csv': str(OUT_COMPONENTS),
        'moves_csv': str(OUT_MOVES),
        'failure_report_cross_split_count': int(failure['cross_split_model_input_fingerprint_count']),
        'plan_applied': False,
        'all_checks_passed': True,
        'elapsed_seconds': time.perf_counter() - started,
    }
    atomic_json(OUT_JSON, summary)

    print()
    print('=' * 92)
    print('MINIMUM-MOVEMENT REPAIR PLAN SUMMARY')
    print('=' * 92)
    print(f'Scaled collision components      : {len(component_rows):,}')
    print(f'Affected original fingerprints   : {summary["affected_original_fingerprint_row_count"]:,}')
    print(f'Minimum fingerprint moves        : {moved_fp:,}')
    print(f'Moved represented raw rows       : {moved_raw:,}')
    print(f'Two-split components             : {summary["component_span_distribution"]["2"]:,}')
    print(f'Three-split components           : {summary["component_span_distribution"]["3"]:,}')
    print()
    print('FAMILY DISTRIBUTION')
    for f in range(3):
        print(f'  family={f} | components={summary["family_distribution"][str(f)]:,}')
    print()
    print('PROPOSED SPLIT DELTAS')
    for row in split_plan:
        print(
            f'  {row["split"]:<10} | fingerprints '
            f'{row["original_fingerprint_count"]:,} {row["fingerprint_delta"]:+,} '
            f'-> {row["proposed_fingerprint_count"]:,} | raw_rows '
            f'{row["original_raw_row_count"]:,} {row["raw_row_delta"]:+,} '
            f'-> {row["proposed_raw_row_count"]:,}'
        )
    print()
    print(f'Summary    : {OUT_JSON}')
    print(f'Components : {OUT_COMPONENTS}')
    print(f'Moves      : {OUT_MOVES}')
    print('Plan applied: False')
    print('All checks passed: True')
    print('SCALED COLLISION REPAIR PLAN COMPLETED')


if __name__ == '__main__':
    main()
