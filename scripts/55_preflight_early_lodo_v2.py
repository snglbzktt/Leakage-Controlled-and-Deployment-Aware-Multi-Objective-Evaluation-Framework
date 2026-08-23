from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

LOCKED_NUMERIC_SUMMARY = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_locked_summary_v2.json"
)

DEVICE_LODO_SUMMARY = (
    AUDIT_DIRECTORY
    / "device_lodo_compact_summary_v2.csv"
)

DEVICE_FAMILY_SUPPORT = (
    AUDIT_DIRECTORY
    / "device_family_support_v2.csv"
)

EXACT_DATABASE = (
    AUDIT_DIRECTORY
    / "nbaiot_exact_byte_audit_v2.sqlite"
)

TASK_DEVICE_AUDIT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "44_audit_task_aware_device_overlap_v2.py"
)

MODEL_REGISTRY = (
    PROJECT_ROOT
    / "models"
    / "architecture"
    / "nbaiot_model_registry_v1.json"
)

OUTPUT_JSON = (
    AUDIT_DIRECTORY
    / "early_lodo_preflight_v2.json"
)

PARQUET_INVENTORY_CSV = (
    AUDIT_DIRECTORY
    / "early_lodo_parquet_inventory_v2.csv"
)

SQLITE_INVENTORY_CSV = (
    AUDIT_DIRECTORY
    / "early_lodo_sqlite_inventory_v2.csv"
)

CODE_HITS_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_device_code_hits_v2.txt"
)

REPORT_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_preflight_v2.txt"
)


EXPECTED_DEVICE_COUNT = 9

EXPECTED_MODELS = {
    "tinyml_mlp",
    "compact_dnn",
}

DEVICE_COLUMN_TERMS = (
    "device",
    "device_name",
    "held_out_device",
    "test_device",
    "source_device",
)

LABEL_COLUMN_TERMS = (
    "label",
    "class",
    "attack",
    "family",
    "target",
)

FINGERPRINT_COLUMN_TERMS = (
    "fingerprint",
    "sha",
    "hash",
    "group",
    "vector",
    "canonical",
)

CODE_PATTERN = re.compile(
    r"(device|held.?out|lodo|leave.?one|"
    r"fingerprint|exact_group|family_3|"
    r"sqlite|select|from|join|csv|parquet)",
    flags=re.IGNORECASE,
)


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
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        rows = list(reader)

        return (
            list(
                reader.fieldnames
                or []
            ),
            rows,
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


def safe_value(
    value: Any,
    maximum_length: int = 120,
) -> str:
    if isinstance(
        value,
        bytes,
    ):
        return (
            f"<bytes length={len(value)}>"
        )

    text = repr(value)

    if len(text) > maximum_length:
        text = (
            text[:maximum_length]
            + "..."
        )

    return text


def find_device_column(
    fieldnames: list[str],
) -> str | None:
    normalized = {
        name.casefold():
            name
        for name in fieldnames
    }

    for candidate in DEVICE_COLUMN_TERMS:
        if candidate in normalized:
            return normalized[candidate]

    for name in fieldnames:
        if "device" in name.casefold():
            return name

    return None


def unique_device_values(
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> tuple[
    str | None,
    set[str],
]:
    column = find_device_column(
        fieldnames
    )

    if column is None:
        return (
            None,
            set(),
        )

    values = {
        str(
            row.get(
                column,
                "",
            )
        ).strip()
        for row in rows
    }

    values.discard("")

    return (
        column,
        values,
    )


def module_status(
    module_name: str,
) -> dict[str, Any]:
    available = (
        importlib.util.find_spec(
            module_name
        )
        is not None
    )

    version = None
    error = None

    if available:
        try:
            module = importlib.import_module(
                module_name
            )

            version = getattr(
                module,
                "__version__",
                None,
            )

        except Exception as exception:
            error = repr(exception)

    return {
        "available":
            available,

        "version":
            version,

        "error":
            error,
    }


def classify_columns(
    columns: list[str],
) -> dict[str, list[str]]:
    device_columns = []

    label_columns = []

    fingerprint_columns = []

    for column in columns:
        normalized = column.casefold()

        if any(
            term in normalized
            for term in DEVICE_COLUMN_TERMS
        ):
            device_columns.append(
                column
            )

        if any(
            term in normalized
            for term in LABEL_COLUMN_TERMS
        ):
            label_columns.append(
                column
            )

        if any(
            term in normalized
            for term
            in FINGERPRINT_COLUMN_TERMS
        ):
            fingerprint_columns.append(
                column
            )

    return {
        "device_columns":
            device_columns,

        "label_columns":
            label_columns,

        "fingerprint_columns":
            fingerprint_columns,
    }


required_files = [
    LOCKED_NUMERIC_SUMMARY,
    DEVICE_LODO_SUMMARY,
    DEVICE_FAMILY_SUPPORT,
    EXACT_DATABASE,
    TASK_DEVICE_AUDIT_SCRIPT,
    MODEL_REGISTRY,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Eksik Faz 2A girdileri:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


numeric_summary = load_json(
    LOCKED_NUMERIC_SUMMARY
)

model_registry = load_json(
    MODEL_REGISTRY
)

(
    lodo_fieldnames,
    lodo_rows,
) = load_csv(
    DEVICE_LODO_SUMMARY
)

(
    support_fieldnames,
    support_rows,
) = load_csv(
    DEVICE_FAMILY_SUPPORT
)


(
    lodo_device_column,
    lodo_devices,
) = unique_device_values(
    lodo_fieldnames,
    lodo_rows,
)

(
    support_device_column,
    support_devices,
) = unique_device_values(
    support_fieldnames,
    support_rows,
)

all_devices = (
    lodo_devices
    | support_devices
)


registry_text = json.dumps(
    model_registry,
    ensure_ascii=False,
)

registered_models = {
    model_name
    for model_name in EXPECTED_MODELS
    if model_name in registry_text
}


package_status = {
    "scikit_learn":
        module_status(
            "sklearn"
        ),

    "lightgbm":
        module_status(
            "lightgbm"
        ),

    "xgboost":
        module_status(
            "xgboost"
        ),

    "pandas":
        module_status(
            "pandas"
        ),

    "numpy":
        module_status(
            "numpy"
        ),

    "torch":
        module_status(
            "torch"
        ),
}


sklearn_estimators = {
    "HistGradientBoostingClassifier":
        False,

    "RandomForestClassifier":
        False,

    "ExtraTreesClassifier":
        False,

    "LogisticRegression":
        False,
}

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    from sklearn.linear_model import (
        LogisticRegression,
    )

    sklearn_estimators[
        "HistGradientBoostingClassifier"
    ] = (
        HistGradientBoostingClassifier
        is not None
    )

    sklearn_estimators[
        "RandomForestClassifier"
    ] = (
        RandomForestClassifier
        is not None
    )

    sklearn_estimators[
        "ExtraTreesClassifier"
    ] = (
        ExtraTreesClassifier
        is not None
    )

    sklearn_estimators[
        "LogisticRegression"
    ] = (
        LogisticRegression
        is not None
    )

except Exception:
    pass


sqlite_rows = []

database_uri = (
    EXACT_DATABASE
    .resolve()
    .as_uri()
    + "?mode=ro"
)

connection = sqlite3.connect(
    database_uri,
    uri=True,
)

try:
    table_names = [
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
    ]

    for table_name in table_names:
        escaped_table_name = (
            table_name.replace(
                "'",
                "''",
            )
        )

        table_info = connection.execute(
            f"PRAGMA table_info('{escaped_table_name}')"
        ).fetchall()

        columns = [
            str(row[1])
            for row in table_info
        ]

        classified = classify_columns(
            columns
        )

        index_rows = connection.execute(
            f"PRAGMA index_list('{escaped_table_name}')"
        ).fetchall()

        index_names = [
            str(row[1])
            for row in index_rows
        ]

        sample_values = []

        try:
            sample_cursor = connection.execute(
                f'SELECT * FROM "{table_name}" LIMIT 2'
            )

            sample_columns = [
                description[0]
                for description
                in sample_cursor.description
            ]

            for sample_row in sample_cursor.fetchall():
                sample_values.append(
                    {
                        sample_columns[index]:
                            safe_value(value)
                        for index, value
                        in enumerate(
                            sample_row
                        )
                    }
                )

        except Exception as exception:
            sample_values = [
                {
                    "sample_error":
                        repr(exception)
                }
            ]

        sqlite_rows.append(
            {
                "table_name":
                    table_name,

                "column_count":
                    len(columns),

                "columns":
                    " | ".join(columns),

                "device_columns":
                    " | ".join(
                        classified[
                            "device_columns"
                        ]
                    ),

                "label_columns":
                    " | ".join(
                        classified[
                            "label_columns"
                        ]
                    ),

                "fingerprint_columns":
                    " | ".join(
                        classified[
                            "fingerprint_columns"
                        ]
                    ),

                "indexes":
                    " | ".join(
                        index_names
                    ),

                "sample":
                    json.dumps(
                        sample_values,
                        ensure_ascii=False,
                    ),
            }
        )

finally:
    connection.close()


write_csv(
    SQLITE_INVENTORY_CSV,
    sqlite_rows,
    [
        "table_name",
        "column_count",
        "columns",
        "device_columns",
        "label_columns",
        "fingerprint_columns",
        "indexes",
        "sample",
    ],
)


parquet_rows = []

parquet_paths = sorted(
    path
    for path in (
        PROJECT_ROOT
        / "data"
    ).rglob("*.parquet")
    if path.is_file()
)

for parquet_path in parquet_paths:
    try:
        parquet_file = pq.ParquetFile(
            parquet_path
        )

        columns = [
            field.name
            for field
            in parquet_file.schema_arrow
        ]

        classified = classify_columns(
            columns
        )

        parquet_rows.append(
            {
                "relative_path":
                    str(
                        parquet_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "row_count":
                    int(
                        parquet_file
                        .metadata
                        .num_rows
                    ),

                "column_count":
                    len(columns),

                "device_columns":
                    " | ".join(
                        classified[
                            "device_columns"
                        ]
                    ),

                "label_columns":
                    " | ".join(
                        classified[
                            "label_columns"
                        ]
                    ),

                "fingerprint_columns":
                    " | ".join(
                        classified[
                            "fingerprint_columns"
                        ]
                    ),

                "all_columns":
                    " | ".join(
                        columns
                    ),
            }
        )

    except Exception as exception:
        parquet_rows.append(
            {
                "relative_path":
                    str(
                        parquet_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "row_count":
                    "",

                "column_count":
                    "",

                "device_columns":
                    "",

                "label_columns":
                    "",

                "fingerprint_columns":
                    "",

                "all_columns":
                    f"ERROR: {exception!r}",
            }
        )


write_csv(
    PARQUET_INVENTORY_CSV,
    parquet_rows,
    [
        "relative_path",
        "row_count",
        "column_count",
        "device_columns",
        "label_columns",
        "fingerprint_columns",
        "all_columns",
    ],
)


script_lines = (
    TASK_DEVICE_AUDIT_SCRIPT
    .read_text(
        encoding="utf-8",
        errors="replace",
    )
    .splitlines()
)

code_hits = []

for line_number, line in enumerate(
    script_lines,
    start=1,
):
    if CODE_PATTERN.search(line):
        code_hits.append(
            f"{line_number:05d}: {line}"
        )


CODE_HITS_FILE.write_text(
    "\n".join(
        code_hits
    ),
    encoding="utf-8",
)


candidate_sqlite_tables = [
    row
    for row in sqlite_rows
    if (
        row["device_columns"]
        or row["label_columns"]
        or row["fingerprint_columns"]
    )
]

candidate_parquets = [
    row
    for row in parquet_rows
    if (
        row["device_columns"]
        or row["fingerprint_columns"]
    )
]


available_ram_gb = (
    psutil.virtual_memory().available
    / (1024 ** 3)
)

free_disk_gb = (
    shutil.disk_usage(
        PROJECT_ROOT
    ).free
    / (1024 ** 3)
)

cpu_logical = (
    os.cpu_count()
    or 1
)

strong_baseline_candidates = []

if package_status[
    "lightgbm"
][
    "available"
]:
    strong_baseline_candidates.append(
        "LightGBM"
    )

if package_status[
    "xgboost"
][
    "available"
]:
    strong_baseline_candidates.append(
        "XGBoost"
    )

if sklearn_estimators[
    "HistGradientBoostingClassifier"
]:
    strong_baseline_candidates.append(
        "HistGradientBoostingClassifier"
    )

if sklearn_estimators[
    "ExtraTreesClassifier"
]:
    strong_baseline_candidates.append(
        "ExtraTreesClassifier"
    )

if sklearn_estimators[
    "RandomForestClassifier"
]:
    strong_baseline_candidates.append(
        "RandomForestClassifier"
    )


validation_checks = {
    "phase_1e_locked":
        numeric_summary.get(
            "status"
        )
        == "locked_complete",

    "pipeline_a_selected":
        numeric_summary.get(
            "selected_pipeline"
        )
        == "pipeline_a",

    "test_not_used_for_numeric_selection":
        numeric_summary.get(
            "selection_used_test_metrics"
        )
        is False,

    "device_summary_not_empty":
        len(lodo_rows) > 0,

    "device_support_not_empty":
        len(support_rows) > 0,

    "exact_device_count":
        len(all_devices)
        == EXPECTED_DEVICE_COUNT,

    "tinyml_and_compact_registered":
        registered_models
        == EXPECTED_MODELS,

    "sqlite_database_readable":
        len(sqlite_rows) > 0,

    "strong_baseline_available":
        len(
            strong_baseline_candidates
        )
        > 0,

    "available_ram_at_least_3_gb":
        available_ram_gb >= 3.0,

    "free_disk_at_least_3_gb":
        free_disk_gb >= 3.0,
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "early_lodo_preflight_v2_1",

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "selected_numeric_pipeline":
        numeric_summary.get(
            "selected_pipeline"
        ),

    "device_lodo_summary": {
        "path":
            str(DEVICE_LODO_SUMMARY),

        "row_count":
            len(lodo_rows),

        "columns":
            lodo_fieldnames,

        "device_column":
            lodo_device_column,
    },

    "device_family_support": {
        "path":
            str(DEVICE_FAMILY_SUPPORT),

        "row_count":
            len(support_rows),

        "columns":
            support_fieldnames,

        "device_column":
            support_device_column,
    },

    "devices":
        sorted(all_devices),

    "device_count":
        len(all_devices),

    "registered_pilot_models":
        sorted(
            registered_models
        ),

    "packages":
        package_status,

    "sklearn_estimators":
        sklearn_estimators,

    "strong_baseline_candidates":
        strong_baseline_candidates,

    "sqlite_database": {
        "path":
            str(EXACT_DATABASE),

        "size_gb":
            (
                EXACT_DATABASE.stat().st_size
                / (1024 ** 3)
            ),

        "table_count":
            len(sqlite_rows),

        "candidate_table_count":
            len(
                candidate_sqlite_tables
            ),
    },

    "parquet_inventory": {
        "file_count":
            len(parquet_rows),

        "candidate_file_count":
            len(candidate_parquets),
    },

    "resources": {
        "python_version":
            platform.python_version(),

        "platform":
            platform.platform(),

        "logical_cpu_count":
            cpu_logical,

        "available_ram_gb":
            available_ram_gb,

        "free_disk_gb":
            free_disk_gb,
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,

    "outputs": {
        "sqlite_inventory":
            str(SQLITE_INVENTORY_CSV),

        "parquet_inventory":
            str(PARQUET_INVENTORY_CSV),

        "code_hits":
            str(CODE_HITS_FILE),

        "report":
            str(REPORT_FILE),
    },
}


OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report_lines = [
    "=" * 78,
    "PHASE 2A EARLY LODO PREFLIGHT",
    "=" * 78,
    "",
    (
        "Selected numeric pipeline : "
        f"{summary['selected_numeric_pipeline']}"
    ),
    (
        "Detected device count     : "
        f"{summary['device_count']}"
    ),
    (
        "Registered models         : "
        + ", ".join(
            summary[
                "registered_pilot_models"
            ]
        )
    ),
    (
        "Baseline candidates       : "
        + ", ".join(
            strong_baseline_candidates
        )
    ),
    (
        "Available RAM GB          : "
        f"{available_ram_gb:.3f}"
    ),
    (
        "Free disk GB              : "
        f"{free_disk_gb:.3f}"
    ),
    "",
    "DEVICES",
]

report_lines.extend(
    f"- {device}"
    for device in sorted(
        all_devices
    )
)

report_lines.extend(
    [
        "",
        "SQLITE CANDIDATE TABLES",
    ]
)

for row in candidate_sqlite_tables:
    report_lines.append(
        (
            f"- {row['table_name']} | "
            f"device={row['device_columns']} | "
            f"label={row['label_columns']} | "
            f"fingerprint={row['fingerprint_columns']}"
        )
    )

report_lines.extend(
    [
        "",
        "PARQUET CANDIDATES",
    ]
)

for row in candidate_parquets:
    report_lines.append(
        (
            f"- {row['relative_path']} | "
            f"rows={row['row_count']} | "
            f"device={row['device_columns']} | "
            f"fingerprint={row['fingerprint_columns']}"
        )
    )

report_lines.extend(
    [
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report_lines.append(
        f"{name}: {passed}"
    )

report_lines.extend(
    [
        "",
        (
            "Preflight passed: "
            f"{all_checks_passed}"
        ),
    ]
)


REPORT_FILE.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print("=" * 78)
print("PHASE 2A EARLY LODO PREFLIGHT")
print("=" * 78)
print(
    "Selected numeric pipeline : "
    f"{summary['selected_numeric_pipeline']}"
)
print(
    "Detected device count     : "
    f"{summary['device_count']}"
)
print(
    "Registered models         : "
    + ", ".join(
        summary[
            "registered_pilot_models"
        ]
    )
)
print(
    "Baseline candidates       : "
    + ", ".join(
        strong_baseline_candidates
    )
)
print(
    "Available RAM GB          : "
    f"{available_ram_gb:.3f}"
)
print(
    "Free disk GB              : "
    f"{free_disk_gb:.3f}"
)

print()
print("DEVICES")

for device in sorted(
    all_devices
):
    print(f"- {device}")

print()
print("DEVICE SUMMARY COLUMNS")
print(
    "LODO summary: "
    + " | ".join(
        lodo_fieldnames
    )
)
print(
    "Family support: "
    + " | ".join(
        support_fieldnames
    )
)

print()
print("SQLITE CANDIDATE TABLES")

for row in candidate_sqlite_tables:
    print(
        f"{row['table_name']} | "
        f"device={row['device_columns']} | "
        f"label={row['label_columns']} | "
        f"fingerprint={row['fingerprint_columns']}"
    )

print()
print("PARQUET CANDIDATES")

for row in candidate_parquets:
    print(
        f"{row['relative_path']} | "
        f"rows={row['row_count']} | "
        f"device={row['device_columns']} | "
        f"fingerprint={row['fingerprint_columns']}"
    )

print()
print("PACKAGE STATUS")

for name, status in (
    package_status.items()
):
    print(
        f"{name}: "
        f"available={status['available']} | "
        f"version={status['version']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"JSON      : {OUTPUT_JSON}")
print(f"Report    : {REPORT_FILE}")
print(f"SQLite CSV: {SQLITE_INVENTORY_CSV}")
print(f"Parquet CSV: {PARQUET_INVENTORY_CSV}")
print(f"Code hits : {CODE_HITS_FILE}")

if not all_checks_passed:
    print()
    print("PHASE 2A EARLY LODO PREFLIGHT FAILED")
    sys.exit(1)

print()
print("PHASE 2A EARLY LODO PREFLIGHT PASSED")
