from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

SELECTED_RUN = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "tinyml_mlp"
    / "p50_qat"
    / "seed_2026"
)

CANDIDATE_FILES = [
    SELECTED_RUN / "metrics.json",
    SELECTED_RUN / "best_int8_checkpoint.pt",
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
    / "manifest.json",
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv",
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv",
]

SEARCH_ROOTS = [
    ROOT / "configs",
    ROOT / "src",
    AUDIT,
    SELECTED_RUN,
]

KEY_PATTERN = re.compile(
    r"(class|label|target|category|attack|benign|mirai|gafgyt)",
    re.IGNORECASE,
)


def short(value: Any, limit: int = 500) -> str:
    text = repr(value)

    if len(text) > limit:
        return text[:limit] + "...[truncated]"

    return text


def normalize_mapping(
    value: dict[Any, Any],
) -> list[str] | None:
    if len(value) != 3:
        return None

    if all(
        isinstance(key, str)
        and isinstance(index, (int, np.integer))
        for key, index in value.items()
    ):
        indexes = {
            int(index)
            for index in value.values()
        }

        if indexes == {0, 1, 2}:
            return [
                next(
                    key
                    for key, index in value.items()
                    if int(index) == position
                )
                for position in range(3)
            ]

    converted: dict[int, str] = {}

    for key, label in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            return None

        if not isinstance(label, str):
            return None

        converted[index] = label

    if set(converted.keys()) == {0, 1, 2}:
        return [
            converted[position]
            for position in range(3)
        ]

    return None


def inspect_value(
    value: Any,
    path: str,
    findings: list[dict[str, Any]],
    key_mentions: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        normalized = normalize_mapping(value)

        if normalized is not None:
            findings.append(
                {
                    "kind": "three_class_mapping",
                    "path": path,
                    "labels": normalized,
                    "raw": short(value),
                }
            )

        for key, child in value.items():
            child_path = (
                f"{path}.{key}"
                if path
                else str(key)
            )

            if KEY_PATTERN.search(str(key)):
                key_mentions.append(
                    {
                        "path": child_path,
                        "value": short(child),
                    }
                )

            inspect_value(
                child,
                child_path,
                findings,
                key_mentions,
            )

    elif isinstance(value, (list, tuple)):
        if (
            len(value) == 3
            and all(
                isinstance(item, str)
                for item in value
            )
        ):
            findings.append(
                {
                    "kind": "three_string_list",
                    "path": path,
                    "labels": list(value),
                    "raw": short(value),
                }
            )

        for index, child in enumerate(value):
            inspect_value(
                child,
                f"{path}[{index}]",
                findings,
                key_mentions,
            )


def inspect_json(
    path: Path,
    findings: list[dict[str, Any]],
    key_mentions: list[dict[str, Any]],
) -> None:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        print(
            f"[JSON READ FAILED] {path}: {error}"
        )
        return

    inspect_value(
        value,
        str(path),
        findings,
        key_mentions,
    )


def inspect_checkpoint(
    path: Path,
    findings: list[dict[str, Any]],
    key_mentions: list[dict[str, Any]],
) -> None:
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        print(
            f"[CHECKPOINT READ FAILED] {path}: {error}"
        )
        return

    inspect_value(
        value,
        str(path),
        findings,
        key_mentions,
    )

    if isinstance(value, dict):
        print()
        print("CHECKPOINT TOP-LEVEL KEYS")
        print("-" * 92)

        for key in value.keys():
            print(key)


def inspect_csv(
    path: Path,
    findings: list[dict[str, Any]],
    key_mentions: list[dict[str, Any]],
) -> None:
    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )
    except Exception as error:
        print(
            f"[CSV READ FAILED] {path}: {error}"
        )
        return

    if not rows:
        return

    headers = list(rows[0].keys())

    for header in headers:
        if KEY_PATTERN.search(header):
            values = sorted(
                {
                    row.get(header, "")
                    for row in rows
                    if row.get(header, "")
                }
            )

            key_mentions.append(
                {
                    "path": f"{path}::{header}",
                    "value": short(values),
                }
            )

            if (
                len(values) == 3
                and all(
                    isinstance(item, str)
                    for item in values
                )
            ):
                findings.append(
                    {
                        "kind": "three_unique_csv_values",
                        "path": f"{path}::{header}",
                        "labels": values,
                        "raw": short(values),
                    }
                )


def inspect_text_file(
    path: Path,
    text_hits: list[dict[str, Any]],
) -> None:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if KEY_PATTERN.search(line):
            text_hits.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "text": line.strip()[:500],
                }
            )


print("=" * 92)
print("PHASE 9 CLASS-LABEL ORDER INSPECTION")
print("=" * 92)

for path in CANDIDATE_FILES:
    print(
        f"{'FOUND' if path.exists() else 'MISSING':<8} {path}"
    )

findings: list[dict[str, Any]] = []
key_mentions: list[dict[str, Any]] = []
text_hits: list[dict[str, Any]] = []

for path in CANDIDATE_FILES:
    if not path.exists():
        continue

    suffix = path.suffix.lower()

    if suffix == ".json":
        inspect_json(
            path,
            findings,
            key_mentions,
        )
    elif suffix == ".pt":
        inspect_checkpoint(
            path,
            findings,
            key_mentions,
        )
    elif suffix == ".csv":
        inspect_csv(
            path,
            findings,
            key_mentions,
        )

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
            ".md",
        }:
            inspect_text_file(
                path,
                text_hits,
            )

print()
print("=" * 92)
print("EXACT THREE-CLASS CANDIDATES")
print("=" * 92)

if findings:
    for index, finding in enumerate(
        findings,
        start=1,
    ):
        print(
            f"[{index}] {finding['kind']}"
        )
        print(
            f"Path   : {finding['path']}"
        )
        print(
            f"Labels : {finding['labels']}"
        )
        print(
            f"Raw    : {finding['raw']}"
        )
        print()
else:
    print("NONE")

print()
print("=" * 92)
print("CLASS/LABEL/TARGET KEY MENTIONS")
print("=" * 92)

if key_mentions:
    for index, mention in enumerate(
        key_mentions[:200],
        start=1,
    ):
        print(
            f"[{index}] {mention['path']} = {mention['value']}"
        )
else:
    print("NONE")

print()
print("=" * 92)
print("SOURCE/CONFIG TEXT HITS")
print("=" * 92)

if text_hits:
    for index, hit in enumerate(
        text_hits[:300],
        start=1,
    ):
        print(
            f"[{index}] {hit['path']}:{hit['line']}"
        )
        print(
            f"    {hit['text']}"
        )
else:
    print("NONE")

print()
print("=" * 92)
print("INSPECTION COMPLETE")
print("=" * 92)
print(
    f"Exact three-class candidates     : {len(findings)}"
)
print(
    f"Structured key mentions          : {len(key_mentions)}"
)
print(
    f"Source/config text hits          : {len(text_hits)}"
)
