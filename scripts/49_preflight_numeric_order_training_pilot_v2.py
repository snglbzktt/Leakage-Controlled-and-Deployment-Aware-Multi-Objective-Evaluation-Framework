from __future__ import annotations

import ast
import csv
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "numeric_order_training_pilot_preflight_v2.json"
)

SYMBOLS_CSV = (
    OUTPUT_DIR
    / "numeric_order_training_pilot_symbols_v2.csv"
)

CODE_HITS_CSV = (
    OUTPUT_DIR
    / "numeric_order_training_pilot_code_hits_v2.csv"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "numeric_order_training_pilot_preflight_v2.txt"
)


def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except ValueError:
        return str(path)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
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


required_artifacts = {
    "primary_train":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nbaiot_primary_seed2026"
        / "train.parquet",

    "primary_validation":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nbaiot_primary_seed2026"
        / "validation.parquet",

    "primary_test":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nbaiot_primary_seed2026"
        / "test.parquet",

    "historical_scaler":
        PROJECT_ROOT
        / "models"
        / "preprocessing"
        / "nbaiot_standard_scaler_seed2026.npz",

    "baseline_trainer":
        PROJECT_ROOT
        / "src"
        / "training"
        / "baseline_trainer.py",

    "data_pipeline":
        PROJECT_ROOT
        / "src"
        / "data"
        / "nbaiot_pipeline.py",

    "b0_script":
        PROJECT_ROOT
        / "scripts"
        / "38_run_b0_matched_leakage_ablation.py",

    "phase_1d_summary":
        PROJECT_ROOT
        / "results"
        / "v2"
        / "audit"
        / "numeric_pipeline_order_summary_v2.json",
}


artifact_status = {
    name: {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": (
            path.stat().st_size
            if path.exists()
            else None
        ),
    }
    for name, path in required_artifacts.items()
}


phase_1d_pilot_required = None

phase_1d_path = required_artifacts[
    "phase_1d_summary"
]

if phase_1d_path.exists():
    phase_1d_summary = json.loads(
        phase_1d_path.read_text(
            encoding="utf-8",
        )
    )

    phase_1d_pilot_required = (
        phase_1d_summary.get(
            "requires_training_pilot"
        )
    )


torch_info: dict[str, Any] = {
    "available": False,
    "version": None,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_names": [],
    "error": None,
}

try:
    import torch

    torch_info["available"] = True
    torch_info["version"] = torch.__version__

    torch_info["cuda_available"] = bool(
        torch.cuda.is_available()
    )

    torch_info["cuda_device_count"] = int(
        torch.cuda.device_count()
    )

    torch_info["cuda_device_names"] = [
        torch.cuda.get_device_name(index)
        for index in range(
            torch.cuda.device_count()
        )
    ]

except Exception as error:
    torch_info["error"] = str(error)


memory_info: dict[str, Any] = {
    "total_gb": None,
    "available_gb": None,
    "error": None,
}

try:
    import psutil

    memory = psutil.virtual_memory()

    memory_info["total_gb"] = round(
        memory.total / (1024 ** 3),
        3,
    )

    memory_info["available_gb"] = round(
        memory.available / (1024 ** 3),
        3,
    )

except Exception as error:
    memory_info["error"] = str(error)


symbol_rows: list[dict[str, Any]] = []
code_hit_rows: list[dict[str, Any]] = []

symbol_pattern = re.compile(
    r"(tiny|compact|baseline|model|train|fit|evaluate|dataset|loader)",
    flags=re.IGNORECASE,
)

code_pattern = re.compile(
    r"(TinyML|Compact|B0|epochs|batch_size|learning_rate|"
    r"weight_decay|patience|CrossEntropyLoss|AdamW?|"
    r"DataLoader|macro_f1|class_weight)",
    flags=re.IGNORECASE,
)


python_files: list[Path] = []

for root_name in ("src", "scripts"):
    root = PROJECT_ROOT / root_name

    if root.exists():
        python_files.extend(
            sorted(
                root.rglob("*.py")
            )
        )


for python_file in python_files:
    try:
        source = python_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tree = ast.parse(
            source,
            filename=str(python_file),
        )

    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if symbol_pattern.search(node.name):
                symbol_rows.append(
                    {
                        "relative_path":
                            relative_path(
                                python_file
                            ),

                        "symbol_type":
                            "class",

                        "symbol_name":
                            node.name,

                        "line_number":
                            node.lineno,

                        "arguments":
                            "",

                        "base_classes":
                            "|".join(
                                ast.unparse(base)
                                for base in node.bases
                            ),
                    }
                )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            if symbol_pattern.search(node.name):
                symbol_rows.append(
                    {
                        "relative_path":
                            relative_path(
                                python_file
                            ),

                        "symbol_type":
                            "function",

                        "symbol_name":
                            node.name,

                        "line_number":
                            node.lineno,

                        "arguments":
                            "|".join(
                                argument.arg
                                for argument
                                in node.args.args
                            ),

                        "base_classes":
                            "",
                    }
                )

    for line_number, line in enumerate(
        source.splitlines(),
        start=1,
    ):
        matches = code_pattern.findall(line)

        if matches:
            code_hit_rows.append(
                {
                    "relative_path":
                        relative_path(
                            python_file
                        ),

                    "line_number":
                        line_number,

                    "keywords":
                        "|".join(
                            sorted(
                                set(matches),
                                key=str.casefold,
                            )
                        ),

                    "code":
                        line.strip(),
                }
            )


write_csv(
    SYMBOLS_CSV,
    symbol_rows,
    [
        "relative_path",
        "symbol_type",
        "symbol_name",
        "line_number",
        "arguments",
        "base_classes",
    ],
)

write_csv(
    CODE_HITS_CSV,
    code_hit_rows,
    [
        "relative_path",
        "line_number",
        "keywords",
        "code",
    ],
)


disk = shutil.disk_usage(PROJECT_ROOT)

all_required_artifacts_exist = all(
    item["exists"]
    for item in artifact_status.values()
)

preflight_passed = (
    all_required_artifacts_exist
    and torch_info["available"]
    and phase_1d_pilot_required is True
)


summary = {
    "protocol":
        "numeric_order_training_pilot_preflight_v2_2",

    "status":
        "completed",

    "completed_at":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "project_root":
        str(PROJECT_ROOT),

    "python_version":
        sys.version.split()[0],

    "platform":
        platform.platform(),

    "cpu_count":
        os.cpu_count(),

    "torch":
        torch_info,

    "memory":
        memory_info,

    "free_disk_gb":
        round(
            disk.free / (1024 ** 3),
            3,
        ),

    "python_file_count":
        len(python_files),

    "interesting_symbol_count":
        len(symbol_rows),

    "code_hit_count":
        len(code_hit_rows),

    "phase_1d_training_pilot_required":
        phase_1d_pilot_required,

    "required_artifacts":
        artifact_status,

    "all_required_artifacts_exist":
        all_required_artifacts_exist,

    "preflight_passed":
        preflight_passed,

    "symbols_csv":
        str(SYMBOLS_CSV),

    "code_hits_csv":
        str(CODE_HITS_CSV),

    "report_txt":
        str(REPORT_TXT),
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report_lines = [
    "=" * 78,
    "PHASE 1E - TRAINING PILOT PREFLIGHT",
    "=" * 78,
    f"Project root: {PROJECT_ROOT}",
    f"Python: {summary['python_version']}",
    f"CPU count: {summary['cpu_count']}",
    f"Torch available: {torch_info['available']}",
    f"Torch version: {torch_info['version']}",
    f"CUDA available: {torch_info['cuda_available']}",
    f"CUDA devices: {torch_info['cuda_device_names']}",
    f"Available RAM GB: {memory_info['available_gb']}",
    f"Free disk GB: {summary['free_disk_gb']}",
    f"Phase 1D pilot required: {phase_1d_pilot_required}",
    "",
    "REQUIRED ARTIFACTS",
    "-" * 78,
]

for name, item in artifact_status.items():
    report_lines.append(
        f"{name}: "
        f"exists={item['exists']} | "
        f"{item['path']}"
    )

report_lines.extend(
    [
        "",
        "MODEL AND TRAINING SYMBOLS",
        "-" * 78,
    ]
)

for row in symbol_rows:
    report_lines.append(
        f"{row['relative_path']}:"
        f"{row['line_number']} | "
        f"{row['symbol_type']} | "
        f"{row['symbol_name']} | "
        f"args={row['arguments']} | "
        f"bases={row['base_classes']}"
    )

report_lines.extend(
    [
        "",
        "VALIDATION",
        "-" * 78,
        (
            "all_required_artifacts_exist: "
            f"{all_required_artifacts_exist}"
        ),
        (
            "torch_available: "
            f"{torch_info['available']}"
        ),
        (
            "phase_1d_training_pilot_required: "
            f"{phase_1d_pilot_required}"
        ),
        (
            "preflight_passed: "
            f"{preflight_passed}"
        ),
    ]
)

REPORT_TXT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


print("=" * 78)
print("PHASE 1E PREFLIGHT SUMMARY")
print("=" * 78)

print(
    "All required artifacts exist : "
    f"{all_required_artifacts_exist}"
)

print(
    "PyTorch available            : "
    f"{torch_info['available']}"
)

print(
    "PyTorch version              : "
    f"{torch_info['version']}"
)

print(
    "CUDA available               : "
    f"{torch_info['cuda_available']}"
)

print(
    "CUDA device names            : "
    f"{torch_info['cuda_device_names']}"
)

print(
    "Available RAM GB             : "
    f"{memory_info['available_gb']}"
)

print(
    "Free disk GB                 : "
    f"{summary['free_disk_gb']}"
)

print(
    "Phase 1D pilot required      : "
    f"{phase_1d_pilot_required}"
)

print(
    "Interesting symbols found    : "
    f"{len(symbol_rows)}"
)

print(
    "Preflight passed             : "
    f"{preflight_passed}"
)

print()
print("MODEL/TRAINING CLASSES")

for row in symbol_rows:
    if row["symbol_type"] == "class":
        print(
            f"{row['relative_path']}:"
            f"{row['line_number']} | "
            f"{row['symbol_name']}"
        )

print()
print("MODEL/TRAINING FUNCTIONS")

for row in symbol_rows:
    if row["symbol_type"] == "function":
        print(
            f"{row['relative_path']}:"
            f"{row['line_number']} | "
            f"{row['symbol_name']}("
            f"{row['arguments']})"
        )

print()
print(f"Summary: {SUMMARY_JSON}")
print(f"Report : {REPORT_TXT}")

if not preflight_passed:
    print()
    print("PHASE 1E PREFLIGHT FAILED")
    sys.exit(1)

print()
print("PHASE 1E PREFLIGHT PASSED")
