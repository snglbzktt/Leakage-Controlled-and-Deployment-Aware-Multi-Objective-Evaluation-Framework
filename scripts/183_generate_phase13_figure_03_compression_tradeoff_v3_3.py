from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
CONFIG = ROOT / "configs" / "protocols"

ASSET_ROOT = ROOT / "results" / "v2" / "manuscript_assets_v3_3"
FIGURE_DIR = ASSET_ROOT / "figures"
SOURCE_DIR = ASSET_ROOT / "source_data"
CAPTION_DIR = ASSET_ROOT / "captions"
METADATA_DIR = ASSET_ROOT / "metadata"

BINDING_PATH = CONFIG / "phase13_exact_source_binding_v3_3.json"
BINDING_LOCK_PATH = AUDIT / "phase13_exact_source_binding_locked_v3_3.json"
MAPPING_PATH = CONFIG / "phase13_manuscript_column_mapping_v3_3.json"
MAPPING_LOCK_PATH = AUDIT / "phase13_manuscript_column_mapping_locked_v3_3.json"

FIGURE_PNG = FIGURE_DIR / "figure_03_compression_tradeoff_v3_3.png"
FIGURE_SVG = FIGURE_DIR / "figure_03_compression_tradeoff_v3_3.svg"
SOURCE_CSV = SOURCE_DIR / "figure_03_compression_tradeoff_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "figure_03_compression_tradeoff_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "figure_03_compression_tradeoff_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_figure_03_compression_tradeoff_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

ARCHITECTURE_LABELS = {
    "tinyml_mlp": "TinyML-MLP",
    "compact_dnn": "Compact-DNN",
}

ARCHITECTURE_MARKERS = {
    "tinyml_mlp": "o",
    "compact_dnn": "s",
}

ANNOTATED_VARIANTS = {
    "B0",
    "DQ",
    "P25-QAT",
    "P50-QAT",
    "P25-FP32-FT",
    "P50-FP32-FT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None
    for attempt in range(1, WINDOWS_FILE_RETRY_COUNT + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            last_error = error
            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break
            time.sleep(WINDOWS_FILE_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: {destination}"
    ) from last_error


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    replace_with_retry(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    replace_with_retry(temporary, path)


def get_bound_source(binding: dict[str, Any], source_id: str) -> Path:
    registry = binding["source_registry"]
    if source_id not in registry:
        raise KeyError(f"Source ID not found in binding: {source_id}")

    record = registry[source_id]
    path = Path(record["path"])

    if not path.exists():
        raise FileNotFoundError(path)

    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Bound source changed after lock: {path}")

    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for output_path in (
    FIGURE_PNG,
    FIGURE_SVG,
    SOURCE_CSV,
    CAPTION_TXT,
    METADATA_JSON,
    LOCK_JSON,
):
    if output_path.exists():
        raise FileExistsError(
            "Figure 03 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)

entry_checks = {
    "binding_locked": (
        binding.get("status") == "locked"
        and binding.get("ready_for_asset_generation") is True
        and binding.get("all_checks_passed") is True
    ),
    "binding_lock_hash_matches": (
        binding_lock["binding"]["sha256"] == sha256_file(BINDING_PATH)
    ),
    "mapping_validated": (
        mapping.get("status") == "validated"
        and mapping.get("ready_for_asset_generation") is True
        and mapping.get("all_checks_passed") is True
    ),
    "mapping_lock_hash_matches": (
        mapping_lock["mapping"]["sha256"] == sha256_file(MAPPING_PATH)
    ),
    "figure_03_mapped": any(
        asset["asset_id"] == "figure_03_compression_tradeoff"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Figure 03 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

phase5_path = get_bound_source(binding, "phase5_group_summary")
phase6_path = get_bound_source(binding, "phase6_group_summary")

phase5_rows = read_csv_rows(phase5_path)
phase6_rows = read_csv_rows(phase6_path)

phase5_lookup = {
    (
        row["architecture"],
        row["variant"],
        row["representation"],
    ): row
    for row in phase5_rows
}

phase6_lookup = {
    (
        row["architecture"],
        row["variant"],
        row["representation"],
    ): row
    for row in phase6_rows
}

phase5_keys = set(phase5_lookup)
phase6_keys = set(phase6_lookup)

data_checks = {
    "phase5_rows_22": len(phase5_rows) == 22,
    "phase6_rows_22": len(phase6_rows) == 22,
    "phase5_unique_22": len(phase5_keys) == 22,
    "phase6_unique_22": len(phase6_keys) == 22,
    "exact_join_key_match": phase5_keys == phase6_keys,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Figure 03 source validation failed: "
        + ", ".join(failed_data_checks)
    )

merged_rows: list[dict[str, Any]] = []

for key in sorted(phase5_keys):
    p5 = phase5_lookup[key]
    p6 = phase6_lookup[key]

    merged_rows.append(
        {
            "architecture": p5["architecture"],
            "architecture_display": ARCHITECTURE_LABELS[p5["architecture"]],
            "variant": p5["variant"],
            "representation": p5["representation"],
            "run_count": int(p5["run_count"]),
            "mean_test_fingerprint_macro_f1": float(
                p5["mean_test_fingerprint_macro_f1"]
            ),
            "std_test_fingerprint_macro_f1": float(
                p5["std_test_fingerprint_macro_f1"]
            ),
            "mean_test_raw_weighted_macro_f1": float(
                p5["mean_test_raw_weighted_macro_f1"]
            ),
            "mean_state_size_ratio_vs_float": float(
                p5["mean_state_size_ratio_vs_float"]
            ),
            "fragility_triggered_in_any_run": (
                p5["fragility_triggered_in_any_run"].strip().lower()
                == "true"
            ),
            "mean_serialized_state_dict_bytes": float(
                p6["mean_serialized_state_dict_bytes"]
            ),
            "mean_state_size_kib": float(
                p6["mean_serialized_state_dict_bytes"]
            ) / 1024.0,
            "mean_peak_inference_RSS_delta_bytes": float(
                p6["mean_peak_inference_RSS_delta_bytes"]
            ),
            "mean_batch_1_median_ms": float(
                p6["mean_batch_1_median_ms"]
            ),
            "mean_batch_32_samples_per_second": float(
                p6["mean_batch_32_samples_per_second"]
            ),
        }
    )

architecture_counts = {
    architecture: sum(
        1
        for row in merged_rows
        if row["architecture"] == architecture
    )
    for architecture in ARCHITECTURE_LABELS
}

post_merge_checks = {
    "merged_rows_22": len(merged_rows) == 22,
    "tinyml_groups_11": architecture_counts["tinyml_mlp"] == 11,
    "compact_groups_11": architecture_counts["compact_dnn"] == 11,
    "all_macro_f1_in_unit_interval": all(
        0.0 <= row["mean_test_fingerprint_macro_f1"] <= 1.0
        for row in merged_rows
    ),
    "all_state_sizes_positive": all(
        row["mean_state_size_kib"] > 0.0
        for row in merged_rows
    ),
    "all_latency_positive": all(
        row["mean_batch_1_median_ms"] > 0.0
        for row in merged_rows
    ),
}

failed_post_merge_checks = [
    name for name, passed in post_merge_checks.items() if not passed
]

if failed_post_merge_checks:
    raise RuntimeError(
        "Figure 03 merged-data checks failed: "
        + ", ".join(failed_post_merge_checks)
    )

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

source_fieldnames = list(merged_rows[0].keys())

with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=source_fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

fig, ax = plt.subplots(figsize=(7.2, 5.1))

for architecture in (
    "tinyml_mlp",
    "compact_dnn",
):
    subset = [
        row
        for row in merged_rows
        if row["architecture"] == architecture
    ]

    x_values = [
        row["mean_state_size_kib"]
        for row in subset
    ]

    y_values = [
        row["mean_test_fingerprint_macro_f1"]
        for row in subset
    ]

    ax.scatter(
        x_values,
        y_values,
        marker=ARCHITECTURE_MARKERS[architecture],
        s=48,
        label=ARCHITECTURE_LABELS[architecture],
    )

    for row in subset:
        if row["variant"] in ANNOTATED_VARIANTS:
            ax.annotate(
                row["variant"],
                (
                    row["mean_state_size_kib"],
                    row["mean_test_fingerprint_macro_f1"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

ax.set_xscale("log")
ax.set_xlabel("Mean serialized state size (KiB, log scale)")
ax.set_ylabel("Mean test fingerprint Macro-F1")
ax.set_title("Compression–Accuracy Trade-off")
ax.grid(True, linewidth=0.5, alpha=0.35)
ax.legend(title="Architecture", frameon=True)

fig.text(
    0.01,
    0.015,
    "Each point is the locked five-seed group mean for one architecture–variant configuration. "
    "Labels are shown for B0 and representative compression/QAT variants.",
    ha="left",
    va="bottom",
    fontsize=7.5,
)

fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(FIGURE_PNG, bbox_inches="tight")
fig.savefig(FIGURE_SVG, bbox_inches="tight")
plt.close(fig)

best_by_architecture = {}

for architecture in (
    "tinyml_mlp",
    "compact_dnn",
):
    subset = [
        row
        for row in merged_rows
        if row["architecture"] == architecture
    ]

    best = max(
        subset,
        key=lambda row: row["mean_test_fingerprint_macro_f1"],
    )

    best_by_architecture[architecture] = {
        "variant": best["variant"],
        "mean_test_fingerprint_macro_f1": (
            best["mean_test_fingerprint_macro_f1"]
        ),
        "mean_state_size_kib": best["mean_state_size_kib"],
    }

caption = (
    "Figure 3. Compression–accuracy trade-off across 22 locked "
    "architecture–variant groups. The x-axis reports mean serialized "
    "state size in KiB on a logarithmic scale, and the y-axis reports "
    "five-seed mean test fingerprint Macro-F1. Circles denote TinyML-MLP "
    "configurations and squares denote Compact-DNN configurations. "
    "The figure is descriptive and does not itself perform model selection; "
    "formal multi-objective eligibility and Pareto analysis are reported "
    "separately in Figure 4."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "figure_03_compression_tradeoff",
    "generated_at_utc": utc_now(),
    "title": "Compression-Accuracy Trade-off",
    "source_files": [
        {
            "source_id": "phase5_group_summary",
            "path": str(phase5_path),
            "sha256": sha256_file(phase5_path),
        },
        {
            "source_id": "phase6_group_summary",
            "path": str(phase6_path),
            "sha256": sha256_file(phase6_path),
        },
    ],
    "outputs": {
        "png": {
            "path": str(FIGURE_PNG),
            "sha256": sha256_file(FIGURE_PNG),
            "size_bytes": FIGURE_PNG.stat().st_size,
        },
        "svg": {
            "path": str(FIGURE_SVG),
            "sha256": sha256_file(FIGURE_SVG),
            "size_bytes": FIGURE_SVG.stat().st_size,
        },
        "source_csv": {
            "path": str(SOURCE_CSV),
            "sha256": sha256_file(SOURCE_CSV),
            "size_bytes": SOURCE_CSV.stat().st_size,
        },
        "caption": {
            "path": str(CAPTION_TXT),
            "sha256": sha256_file(CAPTION_TXT),
            "size_bytes": CAPTION_TXT.stat().st_size,
        },
    },
    "group_count": len(merged_rows),
    "architecture_counts": architecture_counts,
    "best_macro_f1_group_by_architecture": best_by_architecture,
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "post_merge_checks": post_merge_checks,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "figure_03_compression_tradeoff",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "group_count": 22,
    "architecture_count": 2,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 FIGURE 03 - COMPRESSION TRADE-OFF")
print("=" * 96)
print(
    "Merged architecture-variant groups : "
    f"{len(merged_rows)}"
)
print(
    "TinyML-MLP groups                  : "
    f"{architecture_counts['tinyml_mlp']}"
)
print(
    "Compact-DNN groups                 : "
    f"{architecture_counts['compact_dnn']}"
)
print()
print("Highest mean Macro-F1 by architecture:")

for architecture in (
    "tinyml_mlp",
    "compact_dnn",
):
    best = best_by_architecture[architecture]

    print(
        f"  {ARCHITECTURE_LABELS[architecture]:<12} "
        f"{best['variant']:<14} "
        f"Macro-F1={best['mean_test_fingerprint_macro_f1']:.9f} "
        f"State={best['mean_state_size_kib']:.3f} KiB"
    )

print()
print(f"PNG                              : {FIGURE_PNG}")
print(f"SVG                              : {FIGURE_SVG}")
print(f"Source CSV                       : {SOURCE_CSV}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model inference performed        : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 FIGURE 03 GENERATED AND LOCKED")
