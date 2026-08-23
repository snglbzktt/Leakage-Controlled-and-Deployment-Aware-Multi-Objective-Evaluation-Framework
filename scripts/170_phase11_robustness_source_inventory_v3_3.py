from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

CACHE_ROOT = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

PHASE4_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase4_tabular_baselines"
)

PHASE5_B0_REGISTRY = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE9_SOURCE_REGISTRY = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
)

PHASE10_RELEASE_LOCK = (
    AUDIT
    / "phase10_release_archive_locked_v3_3.json"
)

OUTPUT_JSON = (
    AUDIT
    / "phase11_robustness_source_inventory_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase11_robustness_source_inventory_v3_3.txt"
)

SEARCH_ROOTS = (
    ROOT / "results" / "v2",
    ROOT / "models",
    ROOT / "artifacts",
)

MODEL_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
}

HGB_TERMS = (
    "hgb",
    "histgradient",
    "hist_gradient",
    "histogram_gradient",
)


def read_csv_header(
    path: Path,
) -> list[str]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.reader(handle)
        return next(reader)


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def describe_npy(
    path: Path,
) -> dict[str, Any]:
    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    return {
        "path": str(path),
        "name": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size_bytes": int(
            path.stat().st_size
        ),
    }


def safe_json_keys(
    path: Path,
) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        return {
            "path": str(path),
            "error": repr(error),
        }

    if isinstance(value, dict):
        return {
            "path": str(path),
            "top_level_keys": sorted(
                str(key)
                for key in value.keys()
            ),
        }

    return {
        "path": str(path),
        "top_level_type": type(
            value
        ).__name__,
    }


required_paths = (
    CACHE_ROOT,
    PHASE5_B0_REGISTRY,
    PHASE5_MASTER_MATRIX,
    PHASE9_SOURCE_REGISTRY,
    PHASE10_RELEASE_LOCK,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_JSON,
    OUTPUT_TEXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 11 inventory output "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

npy_records = [
    describe_npy(path)
    for path in sorted(
        CACHE_ROOT.rglob("*.npy")
    )
]

npz_records = [
    {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(
            path.stat().st_size
        ),
    }
    for path in sorted(
        CACHE_ROOT.rglob("*.npz")
    )
]

cache_json_records = [
    safe_json_keys(path)
    for path in sorted(
        CACHE_ROOT.rglob("*.json")
    )
]

B0_rows = read_csv_rows(
    PHASE5_B0_REGISTRY
)

B0_seed2026_rows = [
    row
    for row in B0_rows
    if row.get("seed") == "2026"
]

phase9_rows = read_csv_rows(
    PHASE9_SOURCE_REGISTRY
)

model_candidates: list[
    dict[str, Any]
] = []

seen_paths: set[Path] = set()

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        if (
            path.suffix.lower()
            in MODEL_SUFFIXES
            or any(
                term in lower_name
                for term in HGB_TERMS
            )
        ):
            resolved = path.resolve()

            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)

            model_candidates.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "hgb_name_match": any(
                        term in lower_name
                        for term in HGB_TERMS
                    ),
                }
            )

phase4_files: list[
    dict[str, Any]
] = []

if PHASE4_ROOT.exists():
    for path in sorted(
        PHASE4_ROOT.rglob("*")
    ):
        if path.is_file():
            phase4_files.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

inventory = {
    "status": "completed",
    "phase": 11,
    "artifact_name": (
        "robustness_source_inventory"
    ),
    "cache_root": str(
        CACHE_ROOT
    ),
    "cache_npy_files": (
        npy_records
    ),
    "cache_npz_files": (
        npz_records
    ),
    "cache_json_files": (
        cache_json_records
    ),
    "phase5_B0_registry": {
        "path": str(
            PHASE5_B0_REGISTRY
        ),
        "header": read_csv_header(
            PHASE5_B0_REGISTRY
        ),
        "seed_2026_rows": (
            B0_seed2026_rows
        ),
    },
    "phase5_master_matrix": {
        "path": str(
            PHASE5_MASTER_MATRIX
        ),
        "header": read_csv_header(
            PHASE5_MASTER_MATRIX
        ),
    },
    "phase9_source_registry": {
        "path": str(
            PHASE9_SOURCE_REGISTRY
        ),
        "header": read_csv_header(
            PHASE9_SOURCE_REGISTRY
        ),
        "rows": phase9_rows,
    },
    "phase4_root_exists": (
        PHASE4_ROOT.exists()
    ),
    "phase4_files": phase4_files,
    "model_candidates": (
        sorted(
            model_candidates,
            key=lambda row: (
                not row[
                    "hgb_name_match"
                ],
                row["path"],
            ),
        )
    ),
    "release_lock": (
        safe_json_keys(
            PHASE10_RELEASE_LOCK
        )
    ),
    "all_checks_passed": True,
}

OUTPUT_JSON.write_text(
    json.dumps(
        inventory,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

lines: list[str] = []

lines.extend(
    [
        "=" * 92,
        "PHASE 11 ROBUSTNESS SOURCE INVENTORY",
        "=" * 92,
        f"Cache root                      : {CACHE_ROOT}",
        f"NPY files                       : {len(npy_records)}",
        f"NPZ files                       : {len(npz_records)}",
        f"Cache JSON files                : {len(cache_json_records)}",
        f"B0 seed-2026 registry rows      : {len(B0_seed2026_rows)}",
        f"Phase 9 source registry rows    : {len(phase9_rows)}",
        f"Phase 4 root exists             : {PHASE4_ROOT.exists()}",
        f"Phase 4 files                   : {len(phase4_files)}",
        f"Model-like candidates           : {len(model_candidates)}",
        "",
        "CACHE ARRAY INVENTORY",
        "-" * 92,
    ]
)

for record in npy_records:
    lines.append(
        f"{record['name']:<42} "
        f"shape={record['shape']} "
        f"dtype={record['dtype']} "
        f"bytes={record['size_bytes']}"
    )

lines.extend(
    [
        "",
        "PHASE 5 B0 SEED-2026 ROWS",
        "-" * 92,
    ]
)

if B0_seed2026_rows:
    for index, row in enumerate(
        B0_seed2026_rows,
        start=1,
    ):
        lines.append(
            f"[{index}] "
            + json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
else:
    lines.append("NONE")

lines.extend(
    [
        "",
        "PHASE 9 SELECTED SOURCE ROW",
        "-" * 92,
    ]
)

if phase9_rows:
    for index, row in enumerate(
        phase9_rows,
        start=1,
    ):
        lines.append(
            f"[{index}] "
            + json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
else:
    lines.append("NONE")

lines.extend(
    [
        "",
        "HGB / MODEL CANDIDATES",
        "-" * 92,
    ]
)

prioritized_candidates = sorted(
    model_candidates,
    key=lambda row: (
        not row["hgb_name_match"],
        row["path"],
    ),
)

if prioritized_candidates:
    for index, row in enumerate(
        prioritized_candidates[:120],
        start=1,
    ):
        lines.append(
            f"[{index}] "
            f"hgb_match={row['hgb_name_match']} "
            f"bytes={row['size_bytes']} "
            f"{row['path']}"
        )
else:
    lines.append("NONE")

lines.extend(
    [
        "",
        "=" * 92,
        "INVENTORY COMPLETE",
        "=" * 92,
        f"JSON output                     : {OUTPUT_JSON}",
        "All checks passed               : True",
    ]
)

OUTPUT_TEXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(lines)
)
