from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


MODEL_FILE = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

PILOT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "53_run_numeric_order_training_pilot_v2.py"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

FEATURE_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_feature_alignment_v2"
    / "alignment.json"
)

MEMBERSHIP_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
    / "alignment.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_smoke_interface_inspection_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_smoke_interface_inspection_v2.txt"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_signature(
    value: Any,
) -> str:
    try:
        return str(
            inspect.signature(value)
        )

    except Exception as error:
        return (
            f"<signature unavailable: "
            f"{error!r}>"
        )


def read_python_ast_inventory(
    path: Path,
) -> dict[str, Any]:
    source = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    functions = []
    classes = []
    assignments = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                {
                    "name":
                        node.name,

                    "line":
                        node.lineno,

                    "arguments": [
                        argument.arg
                        for argument
                        in (
                            node.args.posonlyargs
                            + node.args.args
                            + node.args.kwonlyargs
                        )
                    ],
                }
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            classes.append(
                {
                    "name":
                        node.name,

                    "line":
                        node.lineno,

                    "methods": [
                        child.name
                        for child
                        in node.body
                        if isinstance(
                            child,
                            (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                            ),
                        )
                    ],
                }
            )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            target_names = []

            targets = (
                node.targets
                if isinstance(
                    node,
                    ast.Assign,
                )
                else [
                    node.target
                ]
            )

            for target in targets:
                if isinstance(
                    target,
                    ast.Name,
                ):
                    target_names.append(
                        target.id
                    )

            assignments.extend(
                {
                    "name":
                        name,

                    "line":
                        node.lineno,
                }
                for name in target_names
            )

    return {
        "functions":
            functions,

        "classes":
            classes,

        "assignments":
            assignments,

        "source_contains_tinyml_mlp":
            "tinyml_mlp"
            in source.casefold(),

        "source_contains_compact_dnn":
            "compact_dnn"
            in source.casefold(),

        "source_contains_model_registry":
            "model_registry"
            in source.casefold(),

        "source_contains_build_model":
            "build_model"
            in source.casefold(),
    }


required_files = [
    MODEL_FILE,
    PILOT_SCRIPT,
    PROTOCOL_FILE,
    FEATURE_ALIGNMENT_FILE,
    MEMBERSHIP_ALIGNMENT_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


protocol = json.loads(
    PROTOCOL_FILE.read_text(
        encoding="utf-8-sig",
    )
)

feature_alignment = json.loads(
    FEATURE_ALIGNMENT_FILE.read_text(
        encoding="utf-8-sig",
    )
)

membership_alignment = json.loads(
    MEMBERSHIP_ALIGNMENT_FILE.read_text(
        encoding="utf-8-sig",
    )
)


model_ast = read_python_ast_inventory(
    MODEL_FILE
)

pilot_ast = read_python_ast_inventory(
    PILOT_SCRIPT
)


module_import_error = None
module_callables = []
module_registries = []

try:
    model_module = importlib.import_module(
        "src.models.nbaiot_models"
    )

    for name, value in inspect.getmembers(
        model_module
    ):
        normalized_name = (
            name.casefold()
        )

        if (
            inspect.isfunction(value)
            or inspect.isclass(value)
        ):
            if (
                value.__module__
                == model_module.__name__
            ):
                module_callables.append(
                    {
                        "name":
                            name,

                        "kind":
                            (
                                "class"
                                if inspect.isclass(
                                    value
                                )
                                else "function"
                            ),

                        "signature":
                            safe_signature(value),
                    }
                )

        if isinstance(value, dict):
            if (
                "registry"
                in normalized_name
                or "model"
                in normalized_name
            ):
                module_registries.append(
                    {
                        "name":
                            name,

                        "keys": [
                            str(key)
                            for key
                            in value.keys()
                        ],

                        "value_types": {
                            str(key):
                                type(item).__name__
                            for key, item
                            in value.items()
                        },
                    }
                )

except Exception as error:
    module_import_error = repr(
        error
    )


feature_splits = (
    feature_alignment.get(
        "splits",
        {}
    )
)

parquet_layout = []

for split in (
    "train",
    "validation",
    "test",
):
    split_entry = feature_splits.get(
        split,
        {},
    )

    feature_path_text = (
        split_entry.get(
            "feature_file"
        )
    )

    if feature_path_text is None:
        parquet_layout.append(
            {
                "split":
                    split,

                "valid":
                    False,

                "error":
                    "feature_file missing",
            }
        )

        continue

    feature_path = Path(
        feature_path_text
    )

    if not feature_path.is_absolute():
        feature_path = (
            PROJECT_ROOT
            / feature_path
        )

    result: dict[str, Any] = {
        "split":
            split,

        "path":
            str(feature_path),

        "valid":
            False,

        "row_count":
            None,

        "column_count":
            None,

        "row_group_count":
            None,

        "row_group_sizes":
            [],

        "feature_count":
            None,

        "error":
            None,
    }

    try:
        parquet_file = pq.ParquetFile(
            feature_path
        )

        schema = (
            parquet_file
            .schema_arrow
        )

        feature_columns = [
            field.name
            for field in schema
            if field.name
            != "class_label"
        ]

        row_group_sizes = [
            int(
                parquet_file
                .metadata
                .row_group(index)
                .num_rows
            )
            for index in range(
                parquet_file
                .metadata
                .num_row_groups
            )
        ]

        result.update(
            {
                "valid":
                    True,

                "row_count":
                    int(
                        parquet_file
                        .metadata
                        .num_rows
                    ),

                "column_count":
                    len(schema),

                "row_group_count":
                    int(
                        parquet_file
                        .metadata
                        .num_row_groups
                    ),

                "row_group_sizes":
                    row_group_sizes,

                "feature_count":
                    len(
                        feature_columns
                    ),
            }
        )

    except Exception as error:
        result["error"] = repr(
            error
        )

    parquet_layout.append(result)


membership_devices = (
    membership_alignment.get(
        "devices",
        []
    )
)

danmini_entries = [
    item
    for item in membership_devices
    if item.get("device")
    == "Danmini_Doorbell"
]


pilot_relevant_functions = [
    item
    for item in pilot_ast[
        "functions"
    ]
    if any(
        term in item["name"].casefold()
        for term in (
            "model",
            "train",
            "evaluate",
            "metric",
            "seed",
            "scaler",
            "loader",
            "dataset",
        )
    )
]


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked",

    "protocol_checks_passed":
        protocol.get(
            "all_checks_passed"
        )
        is True,

    "model_module_imported":
        module_import_error
        is None,

    "model_module_callables_found":
        len(module_callables)
        > 0,

    "tinyml_model_token_found":
        model_ast[
            "source_contains_tinyml_mlp"
        ],

    "compact_model_token_found":
        model_ast[
            "source_contains_compact_dnn"
        ],

    "all_feature_splits_valid":
        len(parquet_layout) == 3
        and all(
            row.get("valid")
            is True
            for row in parquet_layout
        ),

    "all_feature_counts_115":
        all(
            row.get(
                "feature_count"
            )
            == 115
            for row in parquet_layout
        ),

    "danmini_device_found":
        len(danmini_entries)
        == 1,

    "torch_cpu_available":
        isinstance(
            torch.__version__,
            str,
        ),

    "pilot_training_helpers_found":
        len(
            pilot_relevant_functions
        )
        > 0,
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "early_lodo_smoke_interface_inspection_v2_1",

    "status":
        (
            "passed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "model_ast":
        model_ast,

    "pilot_ast":
        pilot_ast,

    "module_import_error":
        module_import_error,

    "module_callables":
        module_callables,

    "module_registries":
        module_registries,

    "pilot_relevant_functions":
        pilot_relevant_functions,

    "parquet_layout":
        parquet_layout,

    "danmini_device":
        (
            danmini_entries[0]
            if danmini_entries
            else None
        ),

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report = [
    "=" * 86,
    "PHASE 2F-0 EARLY LODO SMOKE INTERFACE INSPECTION",
    "=" * 86,
    "",
    "MODEL MODULE API",
]

if module_import_error is not None:
    report.append(
        f"Import error: "
        f"{module_import_error}"
    )

for item in module_callables:
    report.append(
        (
            f"- {item['kind']} "
            f"{item['name']}"
            f"{item['signature']}"
        )
    )


report.extend(
    [
        "",
        "MODEL REGISTRIES",
    ]
)

if module_registries:
    for item in module_registries:
        report.append(
            (
                f"- {item['name']} | "
                f"keys="
                + " | ".join(
                    item["keys"]
                )
            )
        )
else:
    report.append(
        "- No dictionary registry detected."
    )


report.extend(
    [
        "",
        "SCRIPT 53 RELEVANT FUNCTIONS",
    ]
)

for item in pilot_relevant_functions:
    report.append(
        (
            f"- line={item['line']} | "
            f"{item['name']}("
            + ", ".join(
                item["arguments"]
            )
            + ")"
        )
    )


report.extend(
    [
        "",
        "PARQUET LAYOUT",
    ]
)

for row in parquet_layout:
    report.append(
        (
            f"- {row['split']} | "
            f"rows={row.get('row_count')} | "
            f"features="
            f"{row.get('feature_count')} | "
            f"row_groups="
            f"{row.get('row_group_count')} | "
            f"group_sizes="
            f"{row.get('row_group_sizes')} | "
            f"valid={row.get('valid')}"
        )
    )


report.extend(
    [
        "",
        "DANMINI DEVICE",
        json.dumps(
            (
                danmini_entries[0]
                if danmini_entries
                else None
            ),
            ensure_ascii=False,
        ),
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report.append(
        f"{name}: {passed}"
    )


OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("=" * 86)
print("PHASE 2F-0 EARLY LODO SMOKE INTERFACE INSPECTION")
print("=" * 86)

print()
print("MODEL MODULE API")

if module_import_error is not None:
    print(
        f"Import error: "
        f"{module_import_error}"
    )

for item in module_callables:
    print(
        f"{item['kind']} "
        f"{item['name']}"
        f"{item['signature']}"
    )

print()
print("MODEL REGISTRIES")

if module_registries:
    for item in module_registries:
        print(
            f"{item['name']} | "
            + " | ".join(
                item["keys"]
            )
        )
else:
    print(
        "No dictionary registry detected."
    )

print()
print("SCRIPT 53 RELEVANT FUNCTIONS")

for item in pilot_relevant_functions:
    print(
        f"{item['name']}("
        + ", ".join(
            item["arguments"]
        )
        + ")"
    )

print()
print("PARQUET LAYOUT")

for row in parquet_layout:
    print(
        f"{row['split']} | "
        f"rows={row.get('row_count')} | "
        f"features="
        f"{row.get('feature_count')} | "
        f"row_groups="
        f"{row.get('row_group_count')} | "
        f"valid={row.get('valid')}"
    )

print()
print("DANMINI DEVICE")

if danmini_entries:
    print(
        json.dumps(
            danmini_entries[0],
            ensure_ascii=False,
        )
    )
else:
    print("Not found")

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2F-0 SMOKE INTERFACE "
        "INSPECTION FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2F-0 SMOKE INTERFACE "
    "INSPECTION PASSED"
)
