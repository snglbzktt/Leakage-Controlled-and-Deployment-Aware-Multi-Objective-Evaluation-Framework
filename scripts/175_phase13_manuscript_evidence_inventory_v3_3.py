from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE11_LOCK = (
    AUDIT
    / "phase11_input_robustness_locked_v3_3.json"
)

OUTPUT_JSON = (
    AUDIT
    / "phase13_manuscript_evidence_inventory_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase13_manuscript_evidence_inventory_v3_3.txt"
)

SEARCH_SPECS = [
    {
        "section": "phase3_lodo",
        "patterns": [
            "*early_lodo*global*gate*.json",
            "*early_lodo*summary*.json",
            "*phase3*lodo*.json",
            "*phase3*lodo*.csv",
        ],
    },
    {
        "section": "phase4_classical_baselines",
        "patterns": [
            "phase4*.json",
            "phase4*.csv",
            "*tabular*baseline*summary*.json",
            "*hist_gradient_boosting*b0*summary*.json",
            "*logistic_regression*b0*summary*.json",
            "*decision_tree*b0*summary*.json",
            "*random_forest*b0*summary*.json",
        ],
    },
    {
        "section": "phase5_compression",
        "patterns": [
            "phase5_compression_group_summary_v3_2.csv",
            "phase5_compression_paired_delta_vs_B0_v3_2.csv",
            "phase5_compression_master_run_matrix_v3_2.csv",
            "phase5_compression_closure_v3_2.json",
            "phase5_compression_locked_v3_2.json",
        ],
    },
    {
        "section": "phase6_deployment_benchmarks",
        "patterns": [
            "phase6*.json",
            "phase6*.csv",
            "*benchmark*summary*.json",
            "*benchmark*summary*.csv",
        ],
    },
    {
        "section": "phase7_statistics",
        "patterns": [
            "phase7*.json",
            "phase7*.csv",
            "*omnibus*.csv",
            "*posthoc*.csv",
        ],
    },
    {
        "section": "phase8_selection",
        "patterns": [
            "phase8_final_model_family_locked_v3_2.json",
            "phase8*.csv",
            "phase8*.json",
        ],
    },
    {
        "section": "phase9_deployment_artifact",
        "patterns": [
            "phase9_final_deployment_artifact_locked_v3_2.json",
            "phase9_final_artifact_verification_v3_2.json",
            "phase9*.csv",
        ],
    },
    {
        "section": "phase10_release",
        "patterns": [
            "phase10_release_archive_locked_v3_3.json",
            "phase10_release_archive_verification_v3_3.json",
        ],
    },
    {
        "section": "phase11_robustness",
        "patterns": [
            "phase11_input_robustness_locked_v3_3.json",
            "phase11_input_robustness_verification_v3_3.json",
            "phase11_input_robustness_execution_v3_3.json",
            "phase11_input_robustness_completed_v3_3.json",
        ],
    },
]

PHASE11_RESULTS = (
    ROOT
    / "results"
    / "v2"
    / "phase11_input_robustness_v3_3"
)

EXTRA_EXACT_FILES = [
    PHASE11_RESULTS
    / "phase11_input_robustness_all_runs_v3_3.csv",
    PHASE11_RESULTS
    / "phase11_input_robustness_summary_v3_3.json",
    PHASE11_RESULTS
    / "phase11_input_robustness_detailed_results_v3_3.json",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json_summary(
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
            "parse_error": repr(error),
        }

    if not isinstance(value, dict):
        return {
            "top_level_type": type(
                value
            ).__name__,
        }

    important_fields = {}

    for key in (
        "status",
        "phase",
        "artifact_name",
        "protocol_version",
        "all_checks_passed",
        "final_model_changed",
        "ready_for_distribution",
        "ready_for_figures_and_reporting",
        "model_count",
        "condition_count",
        "evaluation_count",
        "verified_evaluation_count",
        "locked_evaluation_count",
        "selected_primary_policy",
        "final_model_family",
        "model_family",
        "representation",
        "canonical_seed",
        "test_fingerprint_macro_f1",
        "test_raw_weighted_macro_f1",
    ):
        if key in value:
            important_fields[key] = value[key]

    return {
        "top_level_keys": sorted(
            str(key)
            for key in value.keys()
        ),
        "important_fields": important_fields,
    }


def read_csv_summary(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        header = list(
            reader.fieldnames
            or []
        )

        row_count = 0
        sample_rows: list[
            dict[str, str]
        ] = []

        for row in reader:
            row_count += 1

            if len(sample_rows) < 3:
                sample_rows.append(row)

    return {
        "header": header,
        "row_count": row_count,
        "sample_rows": sample_rows,
    }


def file_record(
    path: Path,
    section: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "section": section,
        "path": str(path),
        "relative_path": (
            path.relative_to(ROOT)
            .as_posix()
        ),
        "suffix": path.suffix.lower(),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }

    if path.suffix.lower() == ".json":
        record["content_summary"] = (
            read_json_summary(path)
        )

    elif path.suffix.lower() == ".csv":
        record["content_summary"] = (
            read_csv_summary(path)
        )

    return record


if not PHASE11_LOCK.exists():
    raise FileNotFoundError(
        PHASE11_LOCK
    )

phase11_lock = json.loads(
    PHASE11_LOCK.read_text(
        encoding="utf-8"
    )
)

if not (
    phase11_lock.get("status")
    == "locked"
    and phase11_lock.get(
        "phase11_locked"
    )
    is True
    and phase11_lock.get(
        "ready_for_figures_and_reporting"
    )
    is True
    and phase11_lock.get(
        "all_checks_passed"
    )
    is True
):
    raise RuntimeError(
        "Phase 11 is not ready for "
        "figures and reporting."
    )

for output_path in (
    OUTPUT_JSON,
    OUTPUT_TEXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 evidence inventory "
            "already exists; refusing "
            f"to overwrite: {output_path}"
        )

records_by_path: dict[
    Path,
    dict[str, Any],
] = {}

section_counts: dict[
    str,
    int,
] = {}

for search_spec in SEARCH_SPECS:
    section = search_spec["section"]

    matched_paths: set[Path] = set()

    for pattern in search_spec[
        "patterns"
    ]:
        matched_paths.update(
            path
            for path in AUDIT.glob(
                pattern
            )
            if path.is_file()
        )

    for path in sorted(
        matched_paths
    ):
        if path not in records_by_path:
            records_by_path[
                path
            ] = file_record(
                path,
                section,
            )

    section_counts[section] = len(
        matched_paths
    )

for path in EXTRA_EXACT_FILES:
    if not path.exists():
        raise FileNotFoundError(path)

    records_by_path[path] = (
        file_record(
            path,
            "phase11_robustness",
        )
    )

section_counts[
    "phase11_robustness"
] = sum(
    1
    for record in records_by_path.values()
    if record["section"]
    == "phase11_robustness"
)

records = sorted(
    records_by_path.values(),
    key=lambda record: (
        record["section"],
        record["relative_path"],
    ),
)

missing_sections = [
    section
    for section, count
    in section_counts.items()
    if count == 0
]

inventory = {
    "status": "completed",
    "phase": 13,
    "artifact_name": (
        "manuscript_evidence_inventory"
    ),
    "generated_at_utc": utc_now(),
    "phase12_external_dataset": (
        "deferred_optional"
    ),
    "phase11_gate": {
        "path": str(
            PHASE11_LOCK
        ),
        "sha256": sha256_file(
            PHASE11_LOCK
        ),
        "status": (
            phase11_lock.get(
                "status"
            )
        ),
        "ready_for_figures_and_reporting": (
            phase11_lock.get(
                "ready_for_figures_and_reporting"
            )
        ),
    },
    "section_counts": (
        section_counts
    ),
    "missing_sections": (
        missing_sections
    ),
    "file_count": len(records),
    "files": records,
    "model_inference_performed": False,
    "test_data_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_plan": (
        len(missing_sections) == 0
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
    newline="\n",
)

lines: list[str] = []

lines.extend(
    [
        "=" * 92,
        "PHASE 13 MANUSCRIPT EVIDENCE INVENTORY",
        "=" * 92,
        "Phase 12 external dataset       : DEFERRED / OPTIONAL",
        "Phase 11 locked                 : True",
        "Ready for figures/reporting     : True",
        f"Evidence files found           : {len(records)}",
        f"Missing evidence sections      : {len(missing_sections)}",
        "",
        "SECTION COUNTS",
        "-" * 92,
    ]
)

for section, count in (
    section_counts.items()
):
    lines.append(
        f"{section:<36}: {count}"
    )

lines.extend(
    [
        "",
        "EVIDENCE FILES",
        "-" * 92,
    ]
)

for index, record in enumerate(
    records,
    start=1,
):
    summary = record.get(
        "content_summary",
        {},
    )

    if record["suffix"] == ".csv":
        detail = (
            f"rows={summary.get('row_count')} "
            f"columns={len(summary.get('header', []))}"
        )
    else:
        detail = (
            "keys="
            f"{len(summary.get('top_level_keys', []))}"
        )

    lines.append(
        f"[{index:02d}] "
        f"{record['section']:<34} | "
        f"{detail:<20} | "
        f"{record['relative_path']}"
    )

if missing_sections:
    lines.extend(
        [
            "",
            "MISSING SECTIONS",
            "-" * 92,
        ]
    )

    for section in missing_sections:
        lines.append(section)

lines.extend(
    [
        "",
        "=" * 92,
        "INVENTORY COMPLETE",
        "=" * 92,
        f"JSON output                     : {OUTPUT_JSON}",
        "Model inference performed      : False",
        "Test data accessed             : False",
        "Source files mutated           : False",
        "Ready for asset plan           : "
        f"{len(missing_sections) == 0}",
        "All checks passed              : True",
    ]
)

OUTPUT_TEXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
    newline="\n",
)

print(
    "\n".join(lines)
)
