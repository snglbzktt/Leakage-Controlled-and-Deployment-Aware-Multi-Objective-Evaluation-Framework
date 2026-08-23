from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import torch


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

HGB_SUMMARY = (
    AUDIT
    / "hist_gradient_boosting_b0_summary_v3_2.json"
)

HGB_VERIFICATION = (
    AUDIT
    / "hist_gradient_boosting_b0_verification_v3_2.json"
)

CACHE_MANIFEST = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
    / "manifest.json"
)

TINY_B0_CHECKPOINT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "tinyml_mlp"
    / "b0"
    / "seed_2026"
    / "best_checkpoint.pt"
)

TINY_QAT_CHECKPOINT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "tinyml_mlp"
    / "p50_qat"
    / "seed_2026"
    / "best_int8_checkpoint.pt"
)

OUTPUT_JSON = (
    AUDIT
    / "phase11_hgb_and_checkpoint_structure_inspection_v3_3.json"
)

OUTPUT_TEXT = (
    AUDIT
    / "phase11_hgb_and_checkpoint_structure_inspection_v3_3.txt"
)

SEARCH_ROOTS = (
    ROOT / "scripts",
    ROOT / "src",
)

SEARCH_PATTERN = re.compile(
    r"(HistGradientBoostingClassifier|hist_gradient_boosting|"
    r"max_leaf_nodes|max_iter|min_samples_leaf|"
    r"l2_regularization|class_weight|early_stopping)",
    re.IGNORECASE,
)


def read_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def summarize_value(
    value: Any,
) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, dict):
        return {
            str(key): summarize_value(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            summarize_value(child)
            for child in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    return {
        "type": type(value).__name__,
        "repr": repr(value)[:500],
    }


def inspect_checkpoint(
    path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    result: dict[str, Any] = {
        "path": str(path),
        "top_level_type": (
            type(checkpoint).__name__
        ),
    }

    if isinstance(checkpoint, dict):
        result["top_level_keys"] = sorted(
            str(key)
            for key in checkpoint.keys()
        )

        result[
            "summarized_top_level"
        ] = {
            str(key): summarize_value(value)
            for key, value
            in checkpoint.items()
            if key not in {
                "model_state_dict",
                "INT8_model_state_dict",
                "optimizer_state_dict",
            }
        }

        for state_key in (
            "model_state_dict",
            "INT8_model_state_dict",
        ):
            state = checkpoint.get(
                state_key
            )

            if isinstance(state, dict):
                result[
                    state_key
                ] = {
                    "entry_count": len(state),
                    "keys": [
                        str(key)
                        for key in state.keys()
                    ],
                    "entries": {
                        str(key): (
                            summarize_value(value)
                        )
                        for key, value
                        in state.items()
                    },
                }

    return result


def collect_source_hits() -> list[
    dict[str, Any]
]:
    hits: list[dict[str, Any]] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*.py"):
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            matched_lines: list[
                dict[str, Any]
            ] = []

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if SEARCH_PATTERN.search(line):
                    matched_lines.append(
                        {
                            "line_number": (
                                line_number
                            ),
                            "text": (
                                line.strip()[:700]
                            ),
                        }
                    )

            if matched_lines:
                hits.append(
                    {
                        "path": str(path),
                        "matches": (
                            matched_lines[:120]
                        ),
                        "match_count": len(
                            matched_lines
                        ),
                    }
                )

    return hits


def find_non_lodo_hgb_models() -> list[
    dict[str, Any]
]:
    candidates: list[
        dict[str, Any]
    ] = []

    for root in (
        ROOT / "results" / "v2",
        ROOT / "models",
        ROOT / "artifacts",
    ):
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".joblib",
                ".pkl",
                ".pickle",
            }:
                continue

            lower_path = str(
                path
            ).lower()

            if any(
                token in lower_path
                for token in (
                    "early_lodo",
                    "smoke",
                    "invalid_runs",
                )
            ):
                continue

            record: dict[str, Any] = {
                "path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }

            try:
                model = joblib.load(path)
                record[
                    "loaded_type"
                ] = (
                    type(model).__module__
                    + "."
                    + type(model).__name__
                )
            except Exception as error:
                record[
                    "load_error"
                ] = repr(error)

            candidates.append(record)

    return candidates


for path in (
    HGB_SUMMARY,
    HGB_VERIFICATION,
    CACHE_MANIFEST,
    TINY_B0_CHECKPOINT,
    TINY_QAT_CHECKPOINT,
):
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_JSON,
    OUTPUT_TEXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 11 inspection output "
            "already exists; refusing "
            f"to overwrite: {output_path}"
        )

hgb_summary = read_json(
    HGB_SUMMARY
)

hgb_verification = read_json(
    HGB_VERIFICATION
)

cache_manifest = read_json(
    CACHE_MANIFEST
)

B0_inspection = inspect_checkpoint(
    TINY_B0_CHECKPOINT
)

QAT_inspection = inspect_checkpoint(
    TINY_QAT_CHECKPOINT
)

source_hits = collect_source_hits()

non_lodo_hgb_models = (
    find_non_lodo_hgb_models()
)

inspection = {
    "status": "completed",
    "phase": 11,
    "artifact_name": (
        "hgb_and_checkpoint_structure_inspection"
    ),
    "hgb_summary_path": str(
        HGB_SUMMARY
    ),
    "hgb_summary": hgb_summary,
    "hgb_verification_path": str(
        HGB_VERIFICATION
    ),
    "hgb_verification": (
        hgb_verification
    ),
    "cache_manifest_path": str(
        CACHE_MANIFEST
    ),
    "cache_manifest": cache_manifest,
    "tinyml_B0_checkpoint": (
        B0_inspection
    ),
    "tinyml_P50_QAT_checkpoint": (
        QAT_inspection
    ),
    "source_hits": source_hits,
    "non_lodo_hgb_model_candidates": (
        non_lodo_hgb_models
    ),
    "all_checks_passed": True,
}

OUTPUT_JSON.write_text(
    json.dumps(
        inspection,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

lines: list[str] = []

lines.extend(
    [
        "=" * 92,
        "PHASE 11 HGB AND CHECKPOINT STRUCTURE INSPECTION",
        "=" * 92,
        "",
        "HGB SUMMARY",
        "-" * 92,
        json.dumps(
            hgb_summary,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        "",
        "HGB VERIFICATION TOP-LEVEL KEYS",
        "-" * 92,
        json.dumps(
            sorted(
                str(key)
                for key
                in hgb_verification.keys()
            ),
            ensure_ascii=True,
        ),
        "",
        "CACHE MANIFEST TOP-LEVEL KEYS",
        "-" * 92,
        json.dumps(
            sorted(
                str(key)
                for key
                in cache_manifest.keys()
            ),
            ensure_ascii=True,
        ),
        "",
        "TINYML B0 CHECKPOINT",
        "-" * 92,
        json.dumps(
            B0_inspection,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        "",
        "TINYML P50-QAT CHECKPOINT",
        "-" * 92,
        json.dumps(
            QAT_inspection,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        "",
        "NON-LODO HGB MODEL CANDIDATES",
        "-" * 92,
    ]
)

if non_lodo_hgb_models:
    lines.append(
        json.dumps(
            non_lodo_hgb_models,
            indent=2,
            ensure_ascii=True,
        )
    )
else:
    lines.append("NONE")

lines.extend(
    [
        "",
        "HGB SOURCE HITS",
        "-" * 92,
    ]
)

if source_hits:
    for index, record in enumerate(
        source_hits,
        start=1,
    ):
        lines.append(
            f"[{index}] {record['path']} "
            f"(matches={record['match_count']})"
        )

        for match in record[
            "matches"
        ]:
            lines.append(
                f"    L{match['line_number']}: "
                f"{match['text']}"
            )
else:
    lines.append("NONE")

lines.extend(
    [
        "",
        "=" * 92,
        "INSPECTION COMPLETE",
        "=" * 92,
        f"Non-LODO HGB model candidates  : {len(non_lodo_hgb_models)}",
        f"HGB source files with matches  : {len(source_hits)}",
        f"JSON output                     : {OUTPUT_JSON}",
        "All checks passed               : True",
    ]
)

OUTPUT_TEXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(lines)
)
