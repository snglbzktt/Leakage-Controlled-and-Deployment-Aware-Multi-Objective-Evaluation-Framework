from __future__ import annotations

import ast
import csv
import dataclasses
import importlib
import json
import re
import sys
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
    / "training_pilot_protocol_inventory_v2.json"
)

CONSTANTS_CSV = (
    OUTPUT_DIR
    / "training_pilot_protocol_constants_v2.csv"
)

CONFIG_FILES_CSV = (
    OUTPUT_DIR
    / "training_pilot_config_files_v2.csv"
)

SNIPPETS_TXT = (
    OUTPUT_DIR
    / "training_pilot_protocol_snippets_v2.txt"
)


TARGET_FILES = [
    PROJECT_ROOT
    / "src"
    / "models"
    / "nbaiot_models.py",

    PROJECT_ROOT
    / "src"
    / "training"
    / "baseline_trainer.py",

    PROJECT_ROOT
    / "scripts"
    / "22_lock_fp32_baseline_protocol.py",

    PROJECT_ROOT
    / "scripts"
    / "23_run_fp32_baseline_experiments.py",

    PROJECT_ROOT
    / "scripts"
    / "38_run_b0_matched_leakage_ablation.py",
]


NAME_PATTERN = re.compile(
    r"(seed|epoch|batch|learning|lr|weight|patience|"
    r"worker|prefetch|optimizer|scheduler|model|"
    r"architecture|dropout|hidden|class|early|"
    r"tiny|compact|b0)",
    flags=re.IGNORECASE,
)


LINE_PATTERN = re.compile(
    r"(TinyMLMLP|CompactDNN|TinyML|Compact|B0|"
    r"SEEDS?|EPOCHS?|MAX_EPOCHS?|BATCH_SIZE|"
    r"LEARNING_RATE|WEIGHT_DECAY|PATIENCE|"
    r"NUM_WORKERS|PREFETCH_FACTOR|"
    r"early_stopping|AdamW?|CrossEntropyLoss|"
    r"class_weight|macro_f1|train_one|"
    r"run_training|create_model)",
    flags=re.IGNORECASE,
)


CONFIG_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
}


EXCLUDED_PARTS = {
    ".venv",
    "data",
    "results",
    "__pycache__",
    ".git",
}


def relative(path: Path) -> str:
    return str(
        path.relative_to(PROJECT_ROOT)
    )


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


def safe_literal(
    node: ast.AST,
) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__


def extract_constants(
    path: Path,
) -> list[dict[str, Any]]:
    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            else:
                targets = [node.target]
                value_node = node.value

            if value_node is None:
                continue

            for target in targets:
                if not isinstance(
                    target,
                    ast.Name,
                ):
                    continue

                if not NAME_PATTERN.search(
                    target.id
                ):
                    continue

                value = safe_literal(
                    value_node
                )

                rows.append(
                    {
                        "relative_path":
                            relative(path),

                        "line_number":
                            node.lineno,

                        "name":
                            target.id,

                        "value":
                            repr(value),
                    }
                )

    return rows


def extract_function_defaults(
    path: Path,
) -> list[dict[str, Any]]:
    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        if not NAME_PATTERN.search(
            node.name
        ):
            continue

        positional = list(
            node.args.args
        )

        defaults = list(
            node.args.defaults
        )

        missing_count = (
            len(positional)
            - len(defaults)
        )

        aligned_defaults: list[
            ast.AST | None
        ] = (
            [None] * missing_count
            + defaults
        )

        default_parts = []

        for argument, default in zip(
            positional,
            aligned_defaults,
        ):
            if default is not None:
                default_parts.append(
                    f"{argument.arg}="
                    f"{safe_literal(default)!r}"
                )

        for argument, default in zip(
            node.args.kwonlyargs,
            node.args.kw_defaults,
        ):
            if default is not None:
                default_parts.append(
                    f"{argument.arg}="
                    f"{safe_literal(default)!r}"
                )

        if not default_parts:
            continue

        rows.append(
            {
                "relative_path":
                    relative(path),

                "line_number":
                    node.lineno,

                "name":
                    f"function:{node.name}",

                "value":
                    " | ".join(
                        default_parts
                    ),
            }
        )

    return rows


def create_snippets(
    path: Path,
) -> list[str]:
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    matched_indexes = [
        index
        for index, line in enumerate(lines)
        if LINE_PATTERN.search(line)
    ]

    selected_indexes: set[int] = set()

    for index in matched_indexes:
        start = max(
            0,
            index - 2,
        )

        end = min(
            len(lines),
            index + 3,
        )

        selected_indexes.update(
            range(start, end)
        )

    output = [
        "=" * 90,
        relative(path),
        "=" * 90,
    ]

    previous_index: int | None = None

    for index in sorted(
        selected_indexes
    ):
        if (
            previous_index is not None
            and index > previous_index + 1
        ):
            output.append(
                "... omitted ..."
            )

        output.append(
            f"{index + 1:05d}: "
            f"{lines[index]}"
        )

        previous_index = index

    output.append("")

    return output


missing_targets = [
    str(path)
    for path in TARGET_FILES
    if not path.exists()
]

if missing_targets:
    print("Eksik hedef dosyalar:")

    for path in missing_targets:
        print(f"- {path}")

    sys.exit(1)


constant_rows: list[
    dict[str, Any]
] = []

snippet_lines: list[str] = []

for target_file in TARGET_FILES:
    constant_rows.extend(
        extract_constants(
            target_file
        )
    )

    constant_rows.extend(
        extract_function_defaults(
            target_file
        )
    )

    snippet_lines.extend(
        create_snippets(
            target_file
        )
    )


write_csv(
    CONSTANTS_CSV,
    constant_rows,
    [
        "relative_path",
        "line_number",
        "name",
        "value",
    ],
)


config_rows: list[
    dict[str, Any]
] = []

for path in PROJECT_ROOT.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in CONFIG_SUFFIXES:
        continue

    relative_parts = set(
        path.relative_to(
            PROJECT_ROOT
        ).parts
    )

    if relative_parts & EXCLUDED_PARTS:
        continue

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    relevant_lines = [
        line.strip()
        for line in text.splitlines()
        if LINE_PATTERN.search(line)
    ]

    config_rows.append(
        {
            "relative_path":
                relative(path),

            "size_bytes":
                path.stat().st_size,

            "relevant_line_count":
                len(relevant_lines),

            "relevant_lines":
                " || ".join(
                    relevant_lines[:50]
                ),
        }
    )


write_csv(
    CONFIG_FILES_CSV,
    config_rows,
    [
        "relative_path",
        "size_bytes",
        "relevant_line_count",
        "relevant_lines",
    ],
)


model_registry: dict[str, Any] = {
    "import_succeeded": False,
    "registered_models": [],
    "model_details": [],
    "error": None,
}


try:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    module = importlib.import_module(
        "src.models.nbaiot_models"
    )

    registered = list(
        module.list_registered_models()
    )

    model_registry[
        "registered_models"
    ] = registered

    selected_model_names = [
        name
        for name in registered
        if (
            "tinyml" in name.casefold()
            or
            "compact" in name.casefold()
        )
    ]

    for model_name in selected_model_names:
        spec = module.get_model_spec(
            model_name
        )

        if dataclasses.is_dataclass(
            spec
        ):
            spec_data = dataclasses.asdict(
                spec
            )
        elif hasattr(
            spec,
            "__dict__",
        ):
            spec_data = vars(spec)
        else:
            spec_data = repr(spec)

        model = module.create_model(
            model_name=model_name,
            input_features=115,
            num_classes=3,
        )

        total_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )

        model_registry[
            "model_details"
        ].append(
            {
                "model_name":
                    model_name,

                "class_name":
                    model.__class__.__name__,

                "spec":
                    spec_data,

                "total_parameters":
                    total_parameters,

                "trainable_parameters":
                    trainable_parameters,

                "representation":
                    str(model),
            }
        )

    model_registry[
        "import_succeeded"
    ] = True

except Exception as error:
    model_registry[
        "error"
    ] = str(error)


summary = {
    "status":
        "completed",

    "target_files":
        [
            relative(path)
            for path in TARGET_FILES
        ],

    "constant_count":
        len(constant_rows),

    "config_file_count":
        len(config_rows),

    "model_registry":
        model_registry,

    "constants_csv":
        str(CONSTANTS_CSV),

    "config_files_csv":
        str(CONFIG_FILES_CSV),

    "snippets_txt":
        str(SNIPPETS_TXT),
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


SNIPPETS_TXT.write_text(
    "\n".join(
        snippet_lines
    ),
    encoding="utf-8",
)


print("=" * 78)
print("FAZ 1E EĞİTİM PROTOKOLÜ ENVANTERİ")
print("=" * 78)

print()
print("REGISTERED MODELS")

for name in model_registry[
    "registered_models"
]:
    print(f"- {name}")


print()
print("PILOT MODEL DETAILS")

for item in model_registry[
    "model_details"
]:
    print(
        f"{item['model_name']} | "
        f"class={item['class_name']} | "
        f"parameters="
        f"{item['total_parameters']:,}"
    )

    print(
        f"  spec={item['spec']}"
    )


print()
print("IMPORTANT CONSTANTS")

for row in constant_rows:
    print(
        f"{row['relative_path']}:"
        f"{row['line_number']} | "
        f"{row['name']} = "
        f"{row['value']}"
    )


print()
print("CONFIG FILES")

for row in config_rows:
    print(
        f"{row['relative_path']} | "
        f"relevant_lines="
        f"{row['relevant_line_count']}"
    )

    if row[
        "relevant_lines"
    ]:
        print(
            f"  {row['relevant_lines']}"
        )


print()
print(f"Summary : {SUMMARY_JSON}")
print(f"Constants: {CONSTANTS_CSV}")
print(f"Configs : {CONFIG_FILES_CSV}")
print(f"Snippets: {SNIPPETS_TXT}")
print()
print("FAZ 1E PROTOKOL ENVANTERİ TAMAMLANDI")
