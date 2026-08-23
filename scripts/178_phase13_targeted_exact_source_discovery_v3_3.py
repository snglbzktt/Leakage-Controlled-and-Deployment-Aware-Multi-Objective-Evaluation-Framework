from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
RESULTS_V2 = ROOT / "results" / "v2"

SCHEMA_INSPECTION_PATH = (
    AUDIT
    / "phase13_asset_source_schema_inspection_v3_3.json"
)

ASSET_PLAN_LOCK_PATH = (
    AUDIT
    / "phase13_manuscript_asset_plan_locked_v3_3.json"
)

OUTPUT_JSON = (
    AUDIT
    / "phase13_targeted_exact_source_discovery_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase13_targeted_exact_source_discovery_v3_3.txt"
)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize(
    value: str,
) -> str:
    return "".join(
        character
        for character in value.lower()
        if character.isalnum()
    )


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


def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(
        temporary,
        path,
    )


def inspect_csv(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        headers = list(
            reader.fieldnames
            or []
        )

        row_count = 0
        sample_rows: list[
            dict[str, str]
        ] = []

        for row in reader:
            row_count += 1

            if len(sample_rows) < 2:
                sample_rows.append(row)

    return {
        "row_count": row_count,
        "headers": headers,
        "normalized_headers": [
            normalize(header)
            for header in headers
        ],
        "sample_rows": sample_rows,
    }


def find_record_lists(
    value: Any,
    prefix: str = "",
    output: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if output is None:
        output = []

    if isinstance(value, list):
        dict_rows = [
            row
            for row in value
            if isinstance(row, dict)
        ]

        if dict_rows:
            union_keys = sorted(
                {
                    str(key)
                    for row in dict_rows
                    for key in row.keys()
                }
            )

            output.append(
                {
                    "json_path": (
                        prefix or "$"
                    ),
                    "record_count": len(
                        dict_rows
                    ),
                    "keys": union_keys,
                    "normalized_keys": [
                        normalize(key)
                        for key in union_keys
                    ],
                }
            )

        for index, child in enumerate(
            value[:10]
        ):
            find_record_lists(
                child,
                (
                    f"{prefix}[{index}]"
                    if prefix
                    else f"$[{index}]"
                ),
                output,
            )

    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else f"$.{key}"
            )

            find_record_lists(
                child,
                child_prefix,
                output,
            )

    return output


def flatten_keys(
    value: Any,
    prefix: str = "",
    output: list[str] | None = None,
    limit: int = 2000,
) -> list[str]:
    if output is None:
        output = []

    if len(output) >= limit:
        return output

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.append(
                child_prefix
            )

            if len(output) >= limit:
                break

            flatten_keys(
                child,
                child_prefix,
                output,
                limit,
            )

    elif isinstance(value, list):
        for index, child in enumerate(
            value[:10]
        ):
            child_prefix = (
                f"{prefix}[{index}]"
            )

            output.append(
                child_prefix
            )

            if len(output) >= limit:
                break

            flatten_keys(
                child,
                child_prefix,
                output,
                limit,
            )

    return output


def inspect_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        "top_level_type": type(
            value
        ).__name__,
        "top_level_keys": (
            sorted(
                str(key)
                for key in value.keys()
            )
            if isinstance(value, dict)
            else []
        ),
        "flattened_keys": (
            flatten_keys(value)
        ),
        "record_lists": (
            find_record_lists(value)
        ),
    }


def contains_all(
    values: list[str],
    required_groups: list[
        tuple[str, ...]
    ],
) -> bool:
    normalized_values = [
        normalize(value)
        for value in values
    ]

    return all(
        any(
            any(
                token in candidate
                for token in group
            )
            for candidate
            in normalized_values
        )
        for group in required_groups
    )


def score_lodo_candidate(
    path: Path,
    inspection: dict[str, Any],
) -> int:
    path_text = normalize(
        str(path)
    )

    score = 0

    if "earlylodo" in path_text:
        score += 20

    if any(
        token in path_text
        for token in (
            "scientific",
            "final",
            "global",
            "summary",
            "result",
            "locked",
        )
    ):
        score += 10

    if any(
        token in path_text
        for token in (
            "smoke",
            "membership",
            "featurealignment",
            "preflight",
            "invalid",
        )
    ):
        score -= 30

    if path.suffix.lower() == ".csv":
        headers = inspection[
            "headers"
        ]

        if contains_all(
            headers,
            [
                ("device",),
                ("model", "architecture"),
                ("macrof1",),
            ],
        ):
            score += 80

        row_count = int(
            inspection[
                "row_count"
            ]
        )

        if row_count >= 27:
            score += 20

        if row_count == 27:
            score += 10

    elif path.suffix.lower() == ".json":
        flattened_keys = inspection[
            "flattened_keys"
        ]

        if contains_all(
            flattened_keys,
            [
                ("device",),
                ("model", "architecture"),
                ("macrof1",),
            ],
        ):
            score += 60

        for record_list in inspection[
            "record_lists"
        ]:
            keys = record_list["keys"]

            if contains_all(
                keys,
                [
                    ("device",),
                    ("model", "architecture"),
                    ("macrof1",),
                ],
            ):
                score += 80

                if (
                    record_list[
                        "record_count"
                    ]
                    >= 27
                ):
                    score += 20

    return score


def score_omnibus_candidate(
    path: Path,
    inspection: dict[str, Any],
) -> int:
    path_text = normalize(
        str(path)
    )

    score = 0

    if "phase7" in path_text:
        score += 20

    if "omnibus" in path_text:
        score += 60

    if "summary" in path_text:
        score += 10

    if path.suffix.lower() == ".csv":
        headers = inspection[
            "headers"
        ]

        if contains_all(
            headers,
            [
                ("pvalue", "p"),
                ("metric",),
            ],
        ):
            score += 40

        row_count = int(
            inspection[
                "row_count"
            ]
        )

        if row_count == 16:
            score += 30

    else:
        flattened_keys = inspection[
            "flattened_keys"
        ]

        if contains_all(
            flattened_keys,
            [
                ("omnibus",),
                ("pvalue",),
            ],
        ):
            score += 30

    return score


def score_posthoc_candidate(
    path: Path,
    inspection: dict[str, Any],
) -> int:
    path_text = normalize(
        str(path)
    )

    score = 0

    if "phase7" in path_text:
        score += 20

    if "posthoc" in path_text:
        score += 60

    if path.suffix.lower() == ".csv":
        headers = inspection[
            "headers"
        ]

        if contains_all(
            headers,
            [
                ("pvalue",),
                ("holm",),
                ("variant", "comparison"),
            ],
        ):
            score += 40

        row_count = int(
            inspection[
                "row_count"
            ]
        )

        if row_count == 160:
            score += 30

    return score


for path in (
    SCHEMA_INSPECTION_PATH,
    ASSET_PLAN_LOCK_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_JSON,
    OUTPUT_TEXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 targeted source discovery "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

schema_inspection = read_json(
    SCHEMA_INSPECTION_PATH
)

asset_plan_lock = read_json(
    ASSET_PLAN_LOCK_PATH
)

entry_checks = {
    "schema_inspection_completed": (
        schema_inspection.get(
            "status"
        )
        == "completed"
        and schema_inspection.get(
            "ready_for_exact_source_binding"
        )
        is True
        and schema_inspection.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_locked": (
        asset_plan_lock.get(
            "status"
        )
        == "locked"
        and asset_plan_lock.get(
            "ready_for_asset_generation"
        )
        is True
        and asset_plan_lock.get(
            "all_checks_passed"
        )
        is True
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 13 targeted source discovery "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

candidate_paths: set[Path] = set()

for root in (
    AUDIT,
    RESULTS_V2 / "early_lodo",
):
    if not root.exists():
        continue

    for suffix in (
        "*.csv",
        "*.json",
    ):
        candidate_paths.update(
            path
            for path in root.rglob(
                suffix
            )
            if path.is_file()
            and path.stat().st_size
            <= MAX_FILE_SIZE_BYTES
        )

inspections: dict[
    Path,
    dict[str, Any],
] = {}

inspection_errors: list[
    dict[str, str]
] = []

for path in sorted(
    candidate_paths
):
    try:
        if path.suffix.lower() == ".csv":
            inspections[path] = inspect_csv(
                path
            )
        else:
            inspections[path] = inspect_json(
                path
            )
    except Exception as error:
        inspection_errors.append(
            {
                "path": str(path),
                "error": repr(error),
            }
        )

lodo_candidates = []

omnibus_candidates = []

posthoc_candidates = []

for path, inspection in (
    inspections.items()
):
    lodo_score = score_lodo_candidate(
        path,
        inspection,
    )

    if (
        lodo_score > 0
        and "earlylodo"
        in normalize(
            str(path)
        )
    ):
        lodo_candidates.append(
            {
                "score": lodo_score,
                "path": str(path),
                "relative_path": (
                    path.relative_to(
                        ROOT
                    ).as_posix()
                ),
                "suffix": (
                    path.suffix.lower()
                ),
                "sha256": sha256_file(
                    path
                ),
                "inspection": inspection,
            }
        )

    if "phase7" in normalize(
        str(path)
    ):
        omnibus_score = (
            score_omnibus_candidate(
                path,
                inspection,
            )
        )

        if omnibus_score > 0:
            omnibus_candidates.append(
                {
                    "score": omnibus_score,
                    "path": str(path),
                    "relative_path": (
                        path.relative_to(
                            ROOT
                        ).as_posix()
                    ),
                    "suffix": (
                        path.suffix.lower()
                    ),
                    "sha256": (
                        sha256_file(path)
                    ),
                    "inspection": inspection,
                }
            )

        posthoc_score = (
            score_posthoc_candidate(
                path,
                inspection,
            )
        )

        if posthoc_score > 0:
            posthoc_candidates.append(
                {
                    "score": posthoc_score,
                    "path": str(path),
                    "relative_path": (
                        path.relative_to(
                            ROOT
                        ).as_posix()
                    ),
                    "suffix": (
                        path.suffix.lower()
                    ),
                    "sha256": (
                        sha256_file(path)
                    ),
                    "inspection": inspection,
                }
            )

for collection in (
    lodo_candidates,
    omnibus_candidates,
    posthoc_candidates,
):
    collection.sort(
        key=lambda record: (
            -record["score"],
            record["relative_path"],
        )
    )

known_sources = {
    "phase5_group_summary": (
        AUDIT
        / "phase5_compression_group_summary_v3_2.csv"
    ),
    "phase6_group_summary": (
        AUDIT
        / "phase6_deployment_benchmark_group_summary_v3_2.csv"
    ),
    "phase8_input_matrix": (
        AUDIT
        / "phase8_multi_objective_decision_input_matrix_v3_2.csv"
    ),
    "phase8_pareto_results": (
        AUDIT
        / "phase8_multi_objective_pareto_results_v3_2.csv"
    ),
    "phase8_selection_trace": (
        AUDIT
        / "phase8_multi_objective_selection_trace_v3_2.csv"
    ),
    "phase11_all_runs": (
        RESULTS_V2
        / "phase11_input_robustness_v3_3"
        / "phase11_input_robustness_all_runs_v3_3.csv"
    ),
    "phase4_model_summary": (
        AUDIT
        / "phase4_classical_baselines_model_summary_v3_2.csv"
    ),
}

known_source_records = {}

for name, path in (
    known_sources.items()
):
    if not path.exists():
        raise FileNotFoundError(path)

    known_source_records[name] = {
        "path": str(path),
        "relative_path": (
            path.relative_to(
                ROOT
            ).as_posix()
        ),
        "sha256": sha256_file(path),
        "inspection": (
            inspections.get(path)
            or (
                inspect_csv(path)
                if path.suffix.lower()
                == ".csv"
                else inspect_json(path)
            )
        ),
    }

discovery_checks = {
    "lodo_candidates_found": (
        len(lodo_candidates) > 0
    ),
    "phase7_omnibus_candidates_found": (
        len(omnibus_candidates) > 0
    ),
    "phase7_posthoc_candidates_found": (
        len(posthoc_candidates) > 0
    ),
    "phase5_group_summary_rows_22": (
        known_source_records[
            "phase5_group_summary"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 22
    ),
    "phase6_group_summary_rows_22": (
        known_source_records[
            "phase6_group_summary"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 22
    ),
    "phase8_input_matrix_rows_22": (
        known_source_records[
            "phase8_input_matrix"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 22
    ),
    "phase8_selection_trace_rows_3": (
        known_source_records[
            "phase8_selection_trace"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 3
    ),
    "phase11_all_runs_rows_27": (
        known_source_records[
            "phase11_all_runs"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 27
    ),
    "phase4_model_summary_rows_4": (
        known_source_records[
            "phase4_model_summary"
        ][
            "inspection"
        ][
            "row_count"
        ]
        == 4
    ),
}

failed_discovery_checks = [
    name
    for name, passed
    in discovery_checks.items()
    if not passed
]

if failed_discovery_checks:
    raise RuntimeError(
        "Phase 13 targeted discovery checks failed: "
        + ", ".join(
            failed_discovery_checks
        )
    )

result = {
    "status": "completed",
    "phase": 13,
    "artifact_name": (
        "targeted_exact_source_discovery"
    ),
    "generated_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "discovery_checks": (
        discovery_checks
    ),
    "candidate_file_count": (
        len(candidate_paths)
    ),
    "successfully_inspected_file_count": (
        len(inspections)
    ),
    "inspection_errors": (
        inspection_errors
    ),
    "lodo_scientific_candidates": (
        lodo_candidates[:20]
    ),
    "phase7_omnibus_candidates": (
        omnibus_candidates[:20]
    ),
    "phase7_posthoc_candidates": (
        posthoc_candidates[:20]
    ),
    "known_exact_sources": (
        known_source_records
    ),
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "binding_performed": False,
    "ready_for_exact_source_binding": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_JSON,
    result,
)

lines: list[str] = []

lines.extend(
    [
        "=" * 96,
        "PHASE 13 TARGETED EXACT-SOURCE DISCOVERY",
        "=" * 96,
        f"Candidate files scanned         : {len(candidate_paths)}",
        f"Files successfully inspected    : {len(inspections)}",
        f"Inspection errors               : {len(inspection_errors)}",
        "Model inference performed       : False",
        "Test arrays accessed             : False",
        "Source files mutated             : False",
        "",
        "LODO SCIENTIFIC CANDIDATES",
        "-" * 96,
    ]
)

for index, record in enumerate(
    lodo_candidates[:15],
    start=1,
):
    inspection = record[
        "inspection"
    ]

    details = []

    if record["suffix"] == ".csv":
        details.append(
            f"rows={inspection['row_count']}"
        )

        details.append(
            "headers="
            + ",".join(
                inspection[
                    "headers"
                ][:10]
            )
        )
    else:
        record_lists = inspection[
            "record_lists"
        ]

        details.append(
            f"record_lists={len(record_lists)}"
        )

        if record_lists:
            best_record_list = max(
                record_lists,
                key=lambda item: (
                    item[
                        "record_count"
                    ],
                    len(
                        item[
                            "keys"
                        ]
                    ),
                ),
            )

            details.append(
                "largest_list="
                f"{best_record_list['record_count']}"
            )

            details.append(
                "keys="
                + ",".join(
                    best_record_list[
                        "keys"
                    ][:10]
                )
            )

    lines.append(
        f"[{index:02d}] "
        f"score={record['score']:<3} | "
        f"{record['relative_path']}"
    )

    lines.append(
        "     "
        + " | ".join(details)
    )

lines.extend(
    [
        "",
        "PHASE 7 OMNIBUS CANDIDATES",
        "-" * 96,
    ]
)

for index, record in enumerate(
    omnibus_candidates[:10],
    start=1,
):
    inspection = record[
        "inspection"
    ]

    if record["suffix"] == ".csv":
        detail = (
            f"rows={inspection['row_count']} | "
            "headers="
            + ",".join(
                inspection[
                    "headers"
                ][:12]
            )
        )
    else:
        detail = (
            "keys="
            + ",".join(
                inspection[
                    "top_level_keys"
                ][:12]
            )
        )

    lines.append(
        f"[{index:02d}] "
        f"score={record['score']:<3} | "
        f"{record['relative_path']}"
    )

    lines.append(
        f"     {detail}"
    )

lines.extend(
    [
        "",
        "PHASE 7 POST-HOC CANDIDATES",
        "-" * 96,
    ]
)

for index, record in enumerate(
    posthoc_candidates[:10],
    start=1,
):
    inspection = record[
        "inspection"
    ]

    if record["suffix"] == ".csv":
        detail = (
            f"rows={inspection['row_count']} | "
            "headers="
            + ",".join(
                inspection[
                    "headers"
                ][:12]
            )
        )
    else:
        detail = (
            "keys="
            + ",".join(
                inspection[
                    "top_level_keys"
                ][:12]
            )
        )

    lines.append(
        f"[{index:02d}] "
        f"score={record['score']:<3} | "
        f"{record['relative_path']}"
    )

    lines.append(
        f"     {detail}"
    )

lines.extend(
    [
        "",
        "KNOWN EXACT SOURCES",
        "-" * 96,
    ]
)

for name, record in (
    known_source_records.items()
):
    inspection = record[
        "inspection"
    ]

    lines.append(
        f"{name:<34} | "
        f"rows={inspection.get('row_count')} | "
        f"{record['relative_path']}"
    )

lines.extend(
    [
        "",
        "=" * 96,
        "DISCOVERY COMPLETE",
        "=" * 96,
        f"JSON output                     : {OUTPUT_JSON}",
        "Binding performed              : False",
        "Ready for exact source binding : True",
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
