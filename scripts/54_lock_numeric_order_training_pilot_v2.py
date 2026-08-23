from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

RUN_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "numeric_order_training_pilot_v2"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_numeric_order_training_pilot_v2.json"
)

CACHE_SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_pilot_cache_summary_v2.json"
)

SMOKE_SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_smoke_v2.json"
)

RUNS_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_runs_v2.csv"
)

DELTAS_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_paired_deltas_v2.csv"
)

SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_summary_v2.json"
)

LOCKED_SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_locked_summary_v2.json"
)

RELEASE_MANIFEST_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_release_manifest_v2.csv"
)

COMPLETION_DOCUMENT = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "PHASE_1E_NUMERIC_ORDER_TRAINING_PILOT_COMPLETE.md"
)

TRAINING_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "53_run_numeric_order_training_pilot_v2.py"
)

LOCK_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "54_lock_numeric_order_training_pilot_v2.py"
)


EXPECTED_RUN_COUNT = 8
EXPECTED_PAIR_COUNT = 4
EXPECTED_MARGIN = 0.001
EXPECTED_DECISION = (
    "practically_equivalent_retain_pipeline_a"
)
EXPECTED_SELECTED_PIPELINE = "pipeline_a"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


required_files = [
    PROTOCOL_FILE,
    CACHE_SUMMARY_FILE,
    SMOKE_SUMMARY_FILE,
    RUNS_FILE,
    DELTAS_FILE,
    SUMMARY_FILE,
    TRAINING_SCRIPT,
    LOCK_SCRIPT,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Eksik Faz 1E dosyalari:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


protocol = load_json(
    PROTOCOL_FILE
)

cache_summary = load_json(
    CACHE_SUMMARY_FILE
)

smoke_summary = load_json(
    SMOKE_SUMMARY_FILE
)

pilot_summary = load_json(
    SUMMARY_FILE
)

run_rows = load_csv(
    RUNS_FILE
)

delta_rows = load_csv(
    DELTAS_FILE
)


observed_deltas = [
    float(
        row[
            "validation_delta_b_minus_a"
        ]
    )
    for row in delta_rows
]

computed_median_delta = float(
    statistics.median(
        observed_deltas
    )
)

computed_pipeline_b_wins = sum(
    delta > 0.0
    for delta in observed_deltas
)

computed_pipeline_a_wins = sum(
    delta < 0.0
    for delta in observed_deltas
)

all_initializations_match = all(
    str(
        row[
            "initialization_match"
        ]
    ).casefold()
    == "true"
    for row in delta_rows
)

all_test_evaluated_once = all(
    int(
        row[
            "test_evaluation_count"
        ]
    )
    == 1
    for row in run_rows
)

run_ids = {
    row["run_id"]
    for row in run_rows
}

expected_run_ids = {
    str(
        run[
            "run_id"
        ]
    )
    for run in protocol[
        "run_matrix"
    ]
}


validation_checks = {
    "protocol_locked_before_training":
        protocol.get("status")
        == "locked_before_training",

    "protocol_run_count":
        int(
            protocol.get(
                "run_count",
                -1,
            )
        )
        == EXPECTED_RUN_COUNT,

    "cache_checks_passed":
        cache_summary.get(
            "all_checks_passed"
        )
        is True,

    "smoke_checks_passed":
        smoke_summary.get(
            "all_checks_passed"
        )
        is True,

    "pilot_status_completed":
        pilot_summary.get(
            "status"
        )
        == "completed",

    "pilot_checks_passed":
        pilot_summary.get(
            "all_checks_passed"
        )
        is True,

    "run_count":
        len(run_rows)
        == EXPECTED_RUN_COUNT,

    "paired_count":
        len(delta_rows)
        == EXPECTED_PAIR_COUNT,

    "run_matrix_complete":
        run_ids
        == expected_run_ids,

    "all_initializations_match":
        all_initializations_match,

    "all_test_evaluated_once":
        all_test_evaluated_once,

    "median_matches_summary":
        math.isclose(
            computed_median_delta,
            float(
                pilot_summary[
                    "median_validation_delta_b_minus_a"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),

    "pipeline_a_wins_match":
        computed_pipeline_a_wins
        == int(
            pilot_summary[
                "pipeline_a_win_count"
            ]
        ),

    "pipeline_b_wins_match":
        computed_pipeline_b_wins
        == int(
            pilot_summary[
                "pipeline_b_win_count"
            ]
        ),

    "equivalence_margin":
        math.isclose(
            float(
                pilot_summary[
                    "practical_equivalence_margin"
                ]
            ),
            EXPECTED_MARGIN,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),

    "median_inside_equivalence_margin":
        abs(
            computed_median_delta
        )
        < EXPECTED_MARGIN,

    "decision_matches_protocol":
        pilot_summary.get(
            "decision"
        )
        == EXPECTED_DECISION,

    "selected_pipeline_a":
        pilot_summary.get(
            "selected_pipeline"
        )
        == EXPECTED_SELECTED_PIPELINE,

    "test_not_used_for_selection":
        pilot_summary.get(
            "selection_used_test_metrics"
        )
        is False,
}


failed_checks = [
    name
    for name, passed
    in validation_checks.items()
    if not passed
]

if failed_checks:
    print("Faz 1E kilitlenemedi.")
    print("Basarisiz kontroller:")

    for name in failed_checks:
        print(f"- {name}")

    sys.exit(1)


pair_details = []

for row in delta_rows:
    pair_details.append(
        {
            "model_name":
                row["model_name"],

            "seed":
                int(row["seed"]),

            "pipeline_a_validation_macro_f1":
                float(
                    row[
                        "pipeline_a_validation_macro_f1"
                    ]
                ),

            "pipeline_b_validation_macro_f1":
                float(
                    row[
                        "pipeline_b_validation_macro_f1"
                    ]
                ),

            "validation_delta_b_minus_a":
                float(
                    row[
                        "validation_delta_b_minus_a"
                    ]
                ),

            "pipeline_a_test_macro_f1":
                float(
                    row[
                        "pipeline_a_test_macro_f1"
                    ]
                ),

            "pipeline_b_test_macro_f1":
                float(
                    row[
                        "pipeline_b_test_macro_f1"
                    ]
                ),

            "test_delta_b_minus_a":
                float(
                    row[
                        "test_delta_b_minus_a"
                    ]
                ),

            "initialization_match":
                True,

            "winner":
                row["winner"],
        }
    )


locked_summary = {
    "protocol_version":
        "numeric_order_training_pilot_release_v2_1",

    "status":
        "locked_complete",

    "locked_at":
        utc_now(),

    "objective":
        (
            "Determine whether the source-float64 "
            "standardization order materially changes "
            "family-3 model performance."
        ),

    "run_count":
        len(run_rows),

    "paired_count":
        len(delta_rows),

    "models": [
        "tinyml_mlp",
        "compact_dnn",
    ],

    "training_seeds": [
        42,
        123,
    ],

    "computed_validation_deltas_b_minus_a":
        observed_deltas,

    "computed_median_validation_delta_b_minus_a":
        computed_median_delta,

    "pipeline_a_win_count":
        computed_pipeline_a_wins,

    "pipeline_b_win_count":
        computed_pipeline_b_wins,

    "practical_equivalence_margin":
        EXPECTED_MARGIN,

    "decision":
        EXPECTED_DECISION,

    "selected_pipeline":
        EXPECTED_SELECTED_PIPELINE,

    "decision_interpretation":
        (
            "The two numeric processing orders are "
            "practically equivalent under the locked "
            "validation rule. Pipeline A is retained "
            "for continuity with the current model-facing "
            "pipeline and its lower split overlap in the "
            "no-training audit."
        ),

    "selection_used_test_metrics":
        False,

    "pair_details":
        pair_details,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        True,
}


write_json(
    LOCKED_SUMMARY_FILE,
    locked_summary,
)


completion_text = f"""# Phase 1E — Numeric-Order Training Pilot Complete

## Status

Locked and complete.

## Experimental design

- Numeric pipelines: Pipeline A and Pipeline B
- Models: TinyML-MLP and Compact-DNN
- Seeds: 42 and 123
- Total training runs: 8
- Matched comparisons: 4
- Primary selection metric: validation Macro-F1
- Test metrics used for selection: no
- Practical-equivalence margin: ±0.001

## Paired validation results

"""

for pair in pair_details:
    completion_text += (
        f"- `{pair['model_name']}`, seed "
        f"`{pair['seed']}`: "
        f"A={pair['pipeline_a_validation_macro_f1']:.12f}, "
        f"B={pair['pipeline_b_validation_macro_f1']:.12f}, "
        f"B-A={pair['validation_delta_b_minus_a']:+.12f}\n"
    )


completion_text += f"""
## Decision

- Median validation Macro-F1 delta B-A:
  `{computed_median_delta:+.12f}`
- Pipeline A wins: `{computed_pipeline_a_wins}/4`
- Pipeline B wins: `{computed_pipeline_b_wins}/4`
- Locked decision:
  `{EXPECTED_DECISION}`
- Selected pipeline:
  `{EXPECTED_SELECTED_PIPELINE}`

The median difference remains inside the pre-registered
±0.001 practical-equivalence margin. Pipeline A won three
of four matched comparisons, but its median advantage did
not reach the threshold required for a superiority decision.

The two orders are therefore treated as practically
equivalent. Pipeline A is retained for protocol continuity.
Test-set values were recorded but were not used to select
the numeric-processing order.
"""


COMPLETION_DOCUMENT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_DOCUMENT.write_text(
    completion_text,
    encoding="utf-8",
)


manifest_paths = [
    PROTOCOL_FILE,
    CACHE_SUMMARY_FILE,
    SMOKE_SUMMARY_FILE,
    RUNS_FILE,
    DELTAS_FILE,
    SUMMARY_FILE,
    LOCKED_SUMMARY_FILE,
    COMPLETION_DOCUMENT,
    TRAINING_SCRIPT,
    LOCK_SCRIPT,
]

if not RUN_OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Egitim cikti klasoru bulunamadi."
    )

manifest_paths.extend(
    sorted(
        path
        for path in RUN_OUTPUT_ROOT.rglob("*")
        if path.is_file()
    )
)


manifest_rows = []

for path in manifest_paths:
    manifest_rows.append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
    )


write_csv(
    RELEASE_MANIFEST_FILE,
    manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


print("=" * 78)
print("FAZ 1E SAYISAL HAT EGITIM PILOTU")
print("=" * 78)
print(f"Run sayisi                  : {len(run_rows)}")
print(f"Eslesmis karsilastirma      : {len(delta_rows)}")
print(
    "Medyan validation delta B-A: "
    f"{computed_median_delta:+.12f}"
)
print(
    "Pipeline A kazanim sayisi  : "
    f"{computed_pipeline_a_wins}/4"
)
print(
    "Pipeline B kazanim sayisi  : "
    f"{computed_pipeline_b_wins}/4"
)
print(
    "Pratik esdegerlik marji    : "
    f"+/-{EXPECTED_MARGIN}"
)
print(
    "Karar                      : "
    f"{EXPECTED_DECISION}"
)
print(
    "Secilen sayisal hat        : "
    f"{EXPECTED_SELECTED_PIPELINE}"
)
print(
    "Test secimde kullanildi mi : False"
)

print()
print("PAIRED VALIDATION DELTAS")

for pair in pair_details:
    print(
        f"{pair['model_name']} | "
        f"seed={pair['seed']} | "
        f"B-A="
        f"{pair['validation_delta_b_minus_a']:+.12f}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Kilitli ozet : {LOCKED_SUMMARY_FILE}")
print(f"Manifest     : {RELEASE_MANIFEST_FILE}")
print(f"Tamamlama    : {COMPLETION_DOCUMENT}")
print()
print("FAZ 1E SAYISAL HAT EGITIM PILOTU KILITLENDI")
