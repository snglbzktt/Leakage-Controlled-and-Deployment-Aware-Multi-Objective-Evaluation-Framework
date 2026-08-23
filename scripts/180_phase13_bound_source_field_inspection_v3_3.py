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

BINDING_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase13_exact_source_binding_v3_3.json"
)

BINDING_LOCK_PATH = (
    AUDIT
    / "phase13_exact_source_binding_locked_v3_3.json"
)

OUTPUT_JSON = (
    AUDIT
    / "phase13_bound_source_field_inspection_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase13_bound_source_field_inspection_v3_3.txt"
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

PRIMARY_CSV_SOURCE_IDS = [
    "lodo_device_model_matrix",
    "phase4_model_summary",
    "phase5_group_summary",
    "phase5_master_matrix",
    "phase6_group_summary",
    "phase6_locked_matrix",
    "phase7_omnibus",
    "phase7_posthoc_verified",
    "phase8_input_matrix",
    "phase8_pareto_results",
    "phase8_selection_trace",
    "phase11_all_runs",
]

PRIMARY_JSON_SOURCE_IDS = [
    "lodo_global_summary",
    "phase4_summary",
    "phase5_closure",
    "phase6_summary",
    "phase7_summary",
    "phase8_final_model_lock",
    "phase9_artifact_lock",
    "phase10_release_lock",
    "phase11_summary",
    "phase11_lock",
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
        "column_count": len(headers),
        "headers": headers,
        "sample_rows": sample_rows,
    }


def scalar_leaf_paths(
    value: Any,
    prefix: str = "$",
    output: list[dict[str, Any]] | None = None,
    limit: int = 250,
) -> list[dict[str, Any]]:
    if output is None:
        output = []

    if len(output) >= limit:
        return output

    if isinstance(value, dict):
        for key, child in value.items():
            scalar_leaf_paths(
                child,
                f"{prefix}.{key}",
                output,
                limit,
            )

            if len(output) >= limit:
                break

    elif isinstance(value, list):
        for index, child in enumerate(
            value[:8]
        ):
            scalar_leaf_paths(
                child,
                f"{prefix}[{index}]",
                output,
                limit,
            )

            if len(output) >= limit:
                break

    elif isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        output.append(
            {
                "path": prefix,
                "value": value,
                "type": type(
                    value
                ).__name__,
            }
        )

    return output


def inspect_json(
    path: Path,
) -> dict[str, Any]:
    value = read_json(path)

    return {
        "top_level_key_count": len(
            value
        ),
        "top_level_keys": sorted(
            str(key)
            for key in value.keys()
        ),
        "scalar_leaf_paths": (
            scalar_leaf_paths(
                value
            )
        ),
    }


for path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_JSON,
    OUTPUT_TEXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 bound-source field inspection "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(
    BINDING_PATH
)

binding_lock = read_json(
    BINDING_LOCK_PATH
)

entry_checks = {
    "binding_locked": (
        binding.get("status")
        == "locked"
        and binding.get(
            "ready_for_asset_generation"
        )
        is True
        and binding.get(
            "all_checks_passed"
        )
        is True
    ),
    "binding_lock_valid": (
        binding_lock.get("status")
        == "locked"
        and binding_lock.get(
            "ready_for_asset_generation"
        )
        is True
        and binding_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "binding_hash_matches": (
        binding_lock[
            "binding"
        ][
            "sha256"
        ]
        == sha256_file(
            BINDING_PATH
        )
    ),
    "asset_count_12": (
        binding.get(
            "asset_count"
        )
        == 12
    ),
    "unique_source_count_22": (
        binding.get(
            "unique_source_count"
        )
        == 22
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
        "Bound-source field inspection entry "
        "gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

source_registry = binding[
    "source_registry"
]

missing_primary_ids = [
    source_id
    for source_id in (
        PRIMARY_CSV_SOURCE_IDS
        + PRIMARY_JSON_SOURCE_IDS
    )
    if source_id not in source_registry
]

if missing_primary_ids:
    raise RuntimeError(
        "Primary source IDs missing from binding: "
        + ", ".join(
            missing_primary_ids
        )
    )

csv_inspections: dict[
    str,
    dict[str, Any],
] = {}

json_inspections: dict[
    str,
    dict[str, Any],
] = {}

for source_id in (
    PRIMARY_CSV_SOURCE_IDS
):
    record = source_registry[
        source_id
    ]

    path = Path(
        record["path"]
    )

    if not path.exists():
        raise FileNotFoundError(path)

    current_hash = sha256_file(path)

    if current_hash != record[
        "sha256"
    ]:
        raise RuntimeError(
            "Bound CSV changed after lock: "
            f"{path}"
        )

    csv_inspections[
        source_id
    ] = {
        "source_id": source_id,
        "path": str(path),
        "relative_path": (
            record[
                "relative_path"
            ]
        ),
        "sha256": current_hash,
        **inspect_csv(path),
    }

for source_id in (
    PRIMARY_JSON_SOURCE_IDS
):
    record = source_registry[
        source_id
    ]

    path = Path(
        record["path"]
    )

    if not path.exists():
        raise FileNotFoundError(path)

    current_hash = sha256_file(path)

    if current_hash != record[
        "sha256"
    ]:
        raise RuntimeError(
            "Bound JSON changed after lock: "
            f"{path}"
        )

    json_inspections[
        source_id
    ] = {
        "source_id": source_id,
        "path": str(path),
        "relative_path": (
            record[
                "relative_path"
            ]
        ),
        "sha256": current_hash,
        **inspect_json(path),
    }

inspection_checks = {
    "all_primary_csv_sources_inspected": (
        len(csv_inspections)
        == len(
            PRIMARY_CSV_SOURCE_IDS
        )
    ),
    "all_primary_json_sources_inspected": (
        len(json_inspections)
        == len(
            PRIMARY_JSON_SOURCE_IDS
        )
    ),
    "all_csv_sources_have_rows": all(
        record[
            "row_count"
        ]
        > 0
        for record in (
            csv_inspections.values()
        )
    ),
    "all_csv_sources_have_headers": all(
        record[
            "column_count"
        ]
        > 0
        for record in (
            csv_inspections.values()
        )
    ),
    "all_json_sources_have_keys": all(
        record[
            "top_level_key_count"
        ]
        > 0
        for record in (
            json_inspections.values()
        )
    ),
}

failed_inspection_checks = [
    name
    for name, passed
    in inspection_checks.items()
    if not passed
]

if failed_inspection_checks:
    raise RuntimeError(
        "Bound-source field inspection failed: "
        + ", ".join(
            failed_inspection_checks
        )
    )

result = {
    "status": "completed",
    "phase": 13,
    "artifact_name": (
        "bound_source_field_inspection"
    ),
    "generated_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "inspection_checks": (
        inspection_checks
    ),
    "primary_csv_source_count": (
        len(csv_inspections)
    ),
    "primary_json_source_count": (
        len(json_inspections)
    ),
    "csv_sources": (
        csv_inspections
    ),
    "json_sources": (
        json_inspections
    ),
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "asset_generation_performed": False,
    "ready_for_column_mapping": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_JSON,
    result,
)

lines: list[str] = []

lines.extend(
    [
        "=" * 100,
        "PHASE 13 BOUND-SOURCE FIELD INSPECTION",
        "=" * 100,
        f"Primary CSV sources inspected   : {len(csv_inspections)}",
        f"Primary JSON sources inspected  : {len(json_inspections)}",
        "Model inference performed       : False",
        "Test arrays accessed             : False",
        "Source files mutated             : False",
        "Asset generation performed       : False",
        "",
        "PRIMARY CSV SCHEMAS",
        "-" * 100,
    ]
)

for source_id in (
    PRIMARY_CSV_SOURCE_IDS
):
    record = csv_inspections[
        source_id
    ]

    lines.extend(
        [
            source_id.upper(),
            f"Path    : {record['relative_path']}",
            f"Rows    : {record['row_count']}",
            f"Columns : {record['column_count']}",
            "Headers :",
        ]
    )

    for header in record[
        "headers"
    ]:
        lines.append(
            f"  - {header}"
        )

    lines.append("")

lines.extend(
    [
        "PRIMARY JSON SCALAR FIELDS",
        "-" * 100,
    ]
)

for source_id in (
    PRIMARY_JSON_SOURCE_IDS
):
    record = json_inspections[
        source_id
    ]

    lines.extend(
        [
            source_id.upper(),
            f"Path            : {record['relative_path']}",
            f"Top-level keys  : {record['top_level_key_count']}",
            "Scalar leaf paths:",
        ]
    )

    for leaf in record[
        "scalar_leaf_paths"
    ][:120]:
        value = leaf[
            "value"
        ]

        value_text = repr(
            value
        )

        if len(value_text) > 120:
            value_text = (
                value_text[:117]
                + "..."
            )

        lines.append(
            f"  - {leaf['path']} = {value_text}"
        )

    lines.append("")

lines.extend(
    [
        "=" * 100,
        "FIELD INSPECTION COMPLETE",
        "=" * 100,
        f"JSON output                     : {OUTPUT_JSON}",
        "Ready for column mapping        : True",
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
