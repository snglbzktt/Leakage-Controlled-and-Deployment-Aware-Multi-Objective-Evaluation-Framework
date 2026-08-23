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

INVENTORY_PATH = (
    AUDIT
    / "phase13_manuscript_evidence_inventory_v3_3.json"
)

ASSET_PLAN_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase13_manuscript_asset_plan_v3_3.json"
)

ASSET_PLAN_LOCK_PATH = (
    AUDIT
    / "phase13_manuscript_asset_plan_locked_v3_3.json"
)

OUTPUT_JSON = (
    AUDIT
    / "phase13_asset_source_schema_inspection_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase13_asset_source_schema_inspection_v3_3.txt"
)

PROTOCOL_VERSION = (
    "phase13_manuscript_asset_plan_v3_3"
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

ASSET_KEYWORDS = {
    "figure_01_methodology_overview": [
        "protocol",
        "lock",
        "closure",
        "artifact",
        "release",
        "robustness",
        "selection",
    ],
    "figure_02_lodo_generalization": [
        "lodo",
        "device",
        "macro_f1",
        "summary",
        "global",
        "gate",
    ],
    "figure_03_compression_tradeoff": [
        "compression",
        "group_summary",
        "benchmark",
        "latency",
        "throughput",
        "size",
        "macro_f1",
    ],
    "figure_04_pareto_selection": [
        "pareto",
        "selection",
        "phase8",
        "benchmark",
        "latency",
        "macro_f1",
    ],
    "figure_05_input_robustness": [
        "robustness",
        "all_runs",
        "macro_f1",
        "condition",
        "fragility",
    ],
    "table_01_dataset_protocol": [
        "protocol",
        "cache",
        "split",
        "preprocessing",
        "closure",
        "manifest",
    ],
    "table_02_classical_baselines": [
        "baseline",
        "summary",
        "hist_gradient",
        "random_forest",
        "decision_tree",
        "logistic_regression",
    ],
    "table_03_compression_results": [
        "compression",
        "group_summary",
        "master_run_matrix",
        "paired_delta",
    ],
    "table_04_deployment_benchmarks": [
        "benchmark",
        "latency",
        "throughput",
        "phase6",
        "summary",
    ],
    "table_05_statistical_analysis": [
        "phase7",
        "omnibus",
        "posthoc",
        "statistical",
        "holm",
        "permutation",
    ],
    "table_06_robustness_results": [
        "robustness",
        "all_runs",
        "detailed_results",
        "summary",
        "fragility",
    ],
    "table_07_reproducibility_release": [
        "phase8",
        "phase9",
        "phase10",
        "artifact",
        "release",
        "locked",
        "verification",
    ],
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def flatten_json_keys(
    value: Any,
    prefix: str = "",
    limit: int = 600,
) -> list[str]:
    keys: list[str] = []

    def visit(
        current: Any,
        current_prefix: str,
    ) -> None:
        if len(keys) >= limit:
            return

        if isinstance(current, dict):
            for key, child in current.items():
                path = (
                    f"{current_prefix}.{key}"
                    if current_prefix
                    else str(key)
                )

                keys.append(path)

                if len(keys) >= limit:
                    return

                visit(
                    child,
                    path,
                )

        elif isinstance(current, list):
            for index, child in enumerate(
                current[:5]
            ):
                path = (
                    f"{current_prefix}[{index}]"
                )

                keys.append(path)

                if len(keys) >= limit:
                    return

                visit(
                    child,
                    path,
                )

    visit(
        value,
        prefix,
    )

    return keys


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
                sample_rows.append(
                    {
                        key: value
                        for key, value
                        in row.items()
                    }
                )

    return {
        "row_count": row_count,
        "column_count": len(headers),
        "headers": headers,
        "sample_rows": sample_rows,
    }


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
            flatten_json_keys(value)
        ),
    }


def keyword_score(
    text: str,
    keywords: list[str],
) -> int:
    normalized = text.lower()

    score = 0

    for keyword in keywords:
        if keyword.lower() in normalized:
            score += 10

    if normalized.endswith(".csv"):
        score += 4

    elif normalized.endswith(".json"):
        score += 2

    if "locked" in normalized:
        score += 1

    if "verification" in normalized:
        score += 1

    return score


for path in (
    INVENTORY_PATH,
    ASSET_PLAN_PATH,
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
            "Phase 13 source-schema inspection "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

inventory = read_json(
    INVENTORY_PATH
)

asset_plan = read_json(
    ASSET_PLAN_PATH
)

asset_plan_lock = read_json(
    ASSET_PLAN_LOCK_PATH
)

entry_checks = {
    "inventory_completed": (
        inventory.get("status")
        == "completed"
        and inventory.get(
            "ready_for_asset_plan"
        )
        is True
        and inventory.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_locked": (
        asset_plan.get("status")
        == "locked"
        and asset_plan.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and asset_plan.get(
            "ready_for_asset_generation"
        )
        is True
        and asset_plan.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_lock_valid": (
        asset_plan_lock.get("status")
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
    "asset_plan_hash_matches": (
        asset_plan_lock[
            "plan"
        ][
            "sha256"
        ]
        == sha256_file(
            ASSET_PLAN_PATH
        )
    ),
    "asset_count_12": (
        len(
            asset_plan.get(
                "assets",
                [],
            )
        )
        == 12
    ),
    "source_inventory_unchanged": (
        asset_plan[
            "source_inventory"
        ][
            "sha256"
        ]
        == sha256_file(
            INVENTORY_PATH
        )
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
        "Phase 13 source-schema inspection "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

inventory_lookup = {
    record["relative_path"]: record
    for record in inventory[
        "files"
    ]
}

inspected_files: dict[
    str,
    dict[str, Any],
] = {}

asset_candidates: dict[
    str,
    list[dict[str, Any]],
] = {}

for asset in asset_plan["assets"]:
    asset_id = asset[
        "asset_id"
    ]

    keywords = ASSET_KEYWORDS[
        asset_id
    ]

    candidates: list[
        dict[str, Any]
    ] = []

    for evidence in asset[
        "evidence_files"
    ]:
        relative_path = evidence[
            "relative_path"
        ]

        path = (
            ROOT
            / Path(relative_path)
        )

        if not path.exists():
            raise FileNotFoundError(path)

        current_hash = sha256_file(path)

        if current_hash != evidence[
            "sha256"
        ]:
            raise RuntimeError(
                "Evidence file changed after "
                f"asset-plan lock: {path}"
            )

        if relative_path not in inspected_files:
            if path.suffix.lower() == ".csv":
                schema = inspect_csv(path)

            elif path.suffix.lower() == ".json":
                schema = inspect_json(path)

            else:
                schema = {
                    "unsupported_suffix": (
                        path.suffix.lower()
                    )
                }

            inspected_files[
                relative_path
            ] = {
                "path": str(path),
                "relative_path": (
                    relative_path
                ),
                "section": evidence[
                    "section"
                ],
                "suffix": (
                    path.suffix.lower()
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": current_hash,
                "schema": schema,
            }

        record = inspected_files[
            relative_path
        ]

        searchable_text_parts = [
            relative_path,
            evidence["section"],
        ]

        schema = record["schema"]

        if record["suffix"] == ".csv":
            searchable_text_parts.extend(
                schema.get(
                    "headers",
                    [],
                )
            )

        elif record["suffix"] == ".json":
            searchable_text_parts.extend(
                schema.get(
                    "flattened_keys",
                    [],
                )
            )

        score = keyword_score(
            " ".join(
                str(value)
                for value
                in searchable_text_parts
            ),
            keywords,
        )

        candidates.append(
            {
                "relative_path": (
                    relative_path
                ),
                "section": evidence[
                    "section"
                ],
                "suffix": record[
                    "suffix"
                ],
                "score": score,
                "row_count": (
                    schema.get(
                        "row_count"
                    )
                ),
                "column_count": (
                    schema.get(
                        "column_count"
                    )
                ),
                "top_level_key_count": (
                    len(
                        schema.get(
                            "top_level_keys",
                            [],
                        )
                    )
                ),
            }
        )

    candidates.sort(
        key=lambda record: (
            -record["score"],
            record["relative_path"],
        )
    )

    asset_candidates[
        asset_id
    ] = candidates

inspection = {
    "status": "completed",
    "phase": 13,
    "artifact_name": (
        "asset_source_schema_inspection"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "asset_count": 12,
    "inspected_unique_file_count": (
        len(inspected_files)
    ),
    "inspected_files": (
        sorted(
            inspected_files.values(),
            key=lambda record: (
                record["section"],
                record["relative_path"],
            ),
        )
    ),
    "asset_candidates": (
        asset_candidates
    ),
    "ranking_method": (
        "deterministic keyword scoring over locked "
        "relative paths, CSV headers, and JSON key paths"
    ),
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "asset_generation_performed": False,
    "ready_for_exact_source_binding": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_JSON,
    inspection,
)

lines: list[str] = []

lines.extend(
    [
        "=" * 92,
        "PHASE 13 ASSET SOURCE-SCHEMA INSPECTION",
        "=" * 92,
        f"Assets inspected                : {len(asset_candidates)}",
        f"Unique evidence files inspected : {len(inspected_files)}",
        "Model inference performed       : False",
        "Test arrays accessed             : False",
        "Source files mutated             : False",
        "Asset generation performed       : False",
        "",
    ]
)

for asset in asset_plan["assets"]:
    asset_id = asset[
        "asset_id"
    ]

    lines.extend(
        [
            asset_id.upper(),
            "-" * 92,
        ]
    )

    candidates = asset_candidates[
        asset_id
    ][:8]

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        details = []

        if candidate[
            "row_count"
        ] is not None:
            details.append(
                "rows="
                f"{candidate['row_count']}"
            )

        if candidate[
            "column_count"
        ] is not None:
            details.append(
                "cols="
                f"{candidate['column_count']}"
            )

        if candidate[
            "top_level_key_count"
        ] is not None:
            details.append(
                "keys="
                f"{candidate['top_level_key_count']}"
            )

        lines.append(
            f"[{index}] "
            f"score={candidate['score']:<3} | "
            f"{'; '.join(details):<22} | "
            f"{candidate['relative_path']}"
        )

    lines.append("")

lines.extend(
    [
        "=" * 92,
        "INSPECTION COMPLETE",
        "=" * 92,
        f"JSON output                     : {OUTPUT_JSON}",
        "Ready for exact source binding  : True",
        "All checks passed               : True",
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
