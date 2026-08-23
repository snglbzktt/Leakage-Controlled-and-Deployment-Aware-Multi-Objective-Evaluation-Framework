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

FIGURE_04_LOCK = AUDIT / "phase13_figure_04_pareto_selection_locked_v3_3.json"

FIGURE_PNG = FIGURE_DIR / "figure_05_input_robustness_v3_3.png"
FIGURE_SVG = FIGURE_DIR / "figure_05_input_robustness_v3_3.svg"
SOURCE_CSV = SOURCE_DIR / "figure_05_input_robustness_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "figure_05_input_robustness_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "figure_05_input_robustness_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_figure_05_input_robustness_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

CONDITION_ORDER = [
    "clean",
    "gaussian_noise_1pct",
    "gaussian_noise_5pct",
    "gaussian_noise_10pct",
    "feature_mask_5pct",
    "feature_mask_10pct",
    "feature_mask_20pct",
    "scale_drift_minus_10pct",
    "scale_drift_plus_10pct",
]

CONDITION_LABELS = {
    "clean": "Clean",
    "gaussian_noise_1pct": "Noise 1%",
    "gaussian_noise_5pct": "Noise 5%",
    "gaussian_noise_10pct": "Noise 10%",
    "feature_mask_5pct": "Mask 5%",
    "feature_mask_10pct": "Mask 10%",
    "feature_mask_20pct": "Mask 20%",
    "scale_drift_minus_10pct": "Drift -10%",
    "scale_drift_plus_10pct": "Drift +10%",
}

MODEL_ORDER = [
    "tinyml_mlp_B0",
    "tinyml_mlp_P50_QAT",
    "hist_gradient_boosting_B0",
]

MODEL_LABELS = {
    "tinyml_mlp_B0": "TinyML-MLP B0",
    "tinyml_mlp_P50_QAT": "TinyML-MLP P50-QAT",
    "hist_gradient_boosting_B0": "HGB B0",
}

MODEL_MARKERS = {
    "tinyml_mlp_B0": "o",
    "tinyml_mlp_P50_QAT": "s",
    "hist_gradient_boosting_B0": "^",
}

MODEL_LINESTYLES = {
    "tinyml_mlp_B0": "-",
    "tinyml_mlp_P50_QAT": "--",
    "hist_gradient_boosting_B0": "-.",
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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


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


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
    FIGURE_04_LOCK,
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
            "Figure 05 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
figure_04_lock = read_json(FIGURE_04_LOCK)

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
    "figure_04_locked": (
        figure_04_lock.get("status") == "locked"
        and figure_04_lock.get("ready_for_manuscript") is True
        and figure_04_lock.get("all_checks_passed") is True
    ),
    "figure_05_mapped": any(
        asset["asset_id"] == "figure_05_input_robustness"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Figure 05 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

all_runs_path = get_bound_source(binding, "phase11_all_runs")
summary_path = get_bound_source(binding, "phase11_summary")

summary = read_json(summary_path)

rows: list[dict[str, Any]] = []

with all_runs_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    reader = csv.DictReader(handle)

    for raw in reader:
        rows.append(
            {
                "run_id": raw["run_id"],
                "model_id": raw["model_id"],
                "model_display_name": MODEL_LABELS.get(
                    raw["model_id"],
                    raw["model_id"],
                ),
                "condition_id": raw["condition_id"],
                "condition_display_name": CONDITION_LABELS.get(
                    raw["condition_id"],
                    raw["condition_id"],
                ),
                "condition_family": raw["condition_family"],
                "severity": raw["severity"],
                "test_rows": int(raw["test_rows"]),
                "represented_raw_rows": int(raw["represented_raw_rows"]),
                "fingerprint_macro_f1": float(raw["fingerprint_macro_f1"]),
                "raw_weighted_macro_f1": float(raw["raw_weighted_macro_f1"]),
                "macro_f1_drop_from_model_clean": float(
                    raw["macro_f1_drop_from_model_clean"]
                ),
                "macro_f1_retention_ratio": float(
                    raw["macro_f1_retention_ratio"]
                ),
                "severe_fragility": parse_bool(raw["severe_fragility"]),
                "material_sensitivity": parse_bool(
                    raw["material_sensitivity"]
                ),
            }
        )

observed_models = {
    row["model_id"] for row in rows
}

observed_conditions = {
    row["condition_id"] for row in rows
}

data_checks = {
    "row_count_27": len(rows) == 27,
    "model_count_3": len(observed_models) == 3,
    "condition_count_9": len(observed_conditions) == 9,
    "model_set_exact": observed_models == set(MODEL_ORDER),
    "condition_set_exact": observed_conditions == set(CONDITION_ORDER),
    "unique_model_condition_cells": (
        len(
            {
                (row["model_id"], row["condition_id"])
                for row in rows
            }
        )
        == 27
    ),
    "summary_evaluation_count_27": summary["evaluation_count"] == 27,
    "summary_severe_count_5": (
        summary["severe_fragility_evaluation_count"] == 5
    ),
    "summary_material_count_9": (
        summary["material_sensitivity_evaluation_count"] == 9
    ),
    "no_retraining": summary["model_retraining_performed"] is False,
    "test_not_used_for_selection": (
        summary["test_used_for_model_selection"] is False
    ),
    "final_model_unchanged": summary["final_model_changed"] is False,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Figure 05 source validation failed: "
        + ", ".join(failed_data_checks)
    )

lookup = {
    (row["model_id"], row["condition_id"]): row
    for row in rows
}

known_summary_checks = {
    "tinyml_b0_clean": abs(
        lookup[
            ("tinyml_mlp_B0", "clean")
        ]["fingerprint_macro_f1"]
        - summary["model_summaries"]["tinyml_mlp_B0"][
            "clean_fingerprint_macro_f1"
        ]
    ) <= 1e-12,
    "tinyml_p50_qat_clean": abs(
        lookup[
            ("tinyml_mlp_P50_QAT", "clean")
        ]["fingerprint_macro_f1"]
        - summary["model_summaries"]["tinyml_mlp_P50_QAT"][
            "clean_fingerprint_macro_f1"
        ]
    ) <= 1e-12,
    "hgb_clean": abs(
        lookup[
            ("hist_gradient_boosting_B0", "clean")
        ]["fingerprint_macro_f1"]
        - summary["model_summaries"]["hist_gradient_boosting_B0"][
            "clean_fingerprint_macro_f1"
        ]
    ) <= 1e-12,
    "tinyml_b0_worst": (
        summary["model_summaries"]["tinyml_mlp_B0"]["worst_condition"]
        == "feature_mask_20pct"
    ),
    "tinyml_p50_qat_worst": (
        summary["model_summaries"]["tinyml_mlp_P50_QAT"]["worst_condition"]
        == "feature_mask_20pct"
    ),
    "hgb_worst": (
        summary["model_summaries"]["hist_gradient_boosting_B0"][
            "worst_condition"
        ]
        == "scale_drift_plus_10pct"
    ),
}

failed_known_summary_checks = [
    name for name, passed in known_summary_checks.items() if not passed
]

if failed_known_summary_checks:
    raise RuntimeError(
        "Figure 05 locked-summary consistency failed: "
        + ", ".join(failed_known_summary_checks)
    )

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

source_fieldnames = list(rows[0].keys())

with SOURCE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=source_fieldnames,
    )
    writer.writeheader()

    for model_id in MODEL_ORDER:
        for condition_id in CONDITION_ORDER:
            writer.writerow(
                lookup[
                    (model_id, condition_id)
                ]
            )

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

fig, ax = plt.subplots(figsize=(7.5, 5.1))

x_values = list(range(len(CONDITION_ORDER)))

for model_id in MODEL_ORDER:
    y_values = [
        lookup[
            (model_id, condition_id)
        ]["fingerprint_macro_f1"]
        for condition_id in CONDITION_ORDER
    ]

    ax.plot(
        x_values,
        y_values,
        marker=MODEL_MARKERS[model_id],
        linestyle=MODEL_LINESTYLES[model_id],
        linewidth=1.5,
        markersize=5,
        label=MODEL_LABELS[model_id],
    )

    severe_points = [
        (
            index,
            lookup[
                (model_id, condition_id)
            ]["fingerprint_macro_f1"],
        )
        for index, condition_id in enumerate(CONDITION_ORDER)
        if lookup[
            (model_id, condition_id)
        ]["severe_fragility"]
    ]

    for x_value, y_value in severe_points:
        ax.annotate(
            "Severe",
            (x_value, y_value),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

ax.set_xticks(x_values)
ax.set_xticklabels(
    [
        CONDITION_LABELS[
            condition_id
        ]
        for condition_id in CONDITION_ORDER
    ],
    rotation=28,
    ha="right",
)

ax.set_ylim(0.40, 1.02)
ax.set_ylabel("Fingerprint Macro-F1")
ax.set_xlabel("Input condition")
ax.set_title("Input Robustness Under Locked Perturbations")
ax.grid(True, linewidth=0.5, alpha=0.35)
ax.legend(frameon=True)

fig.text(
    0.01,
    0.015,
    "Twenty-seven locked evaluations: three models × nine conditions. "
    "Five severe-fragility evaluations occurred, all for HGB; "
    "no model retraining or final-model change was performed.",
    ha="left",
    va="bottom",
    fontsize=7.5,
)

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURE_PNG, bbox_inches="tight")
fig.savefig(FIGURE_SVG, bbox_inches="tight")
plt.close(fig)

caption = (
    "Figure 5. Input robustness across nine locked perturbation conditions for "
    "TinyML-MLP B0, the selected TinyML-MLP P50-QAT model family, and HGB B0. "
    "TinyML-MLP B0 and P50-QAT retained substantially higher Macro-F1 under the "
    "tested perturbations, whereas HGB showed pronounced sensitivity to Gaussian "
    "noise and scale drift. Five severe-fragility evaluations were observed, all "
    "for HGB. This analysis did not retrain models, did not use test results for "
    "model selection, and did not alter the locked final model."
)

atomic_text(CAPTION_TXT, caption + "\n")

model_summary_export = {}

for model_id in MODEL_ORDER:
    summary_key = model_id

    model_summary_export[model_id] = {
        "clean_fingerprint_macro_f1": summary[
            "model_summaries"
        ][summary_key]["clean_fingerprint_macro_f1"],
        "worst_condition": summary[
            "model_summaries"
        ][summary_key]["worst_condition"],
        "worst_fingerprint_macro_f1": summary[
            "model_summaries"
        ][summary_key]["worst_fingerprint_macro_f1"],
        "maximum_macro_f1_drop": summary[
            "model_summaries"
        ][summary_key]["maximum_macro_f1_drop"],
        "severe_fragility_condition_count": summary[
            "model_summaries"
        ][summary_key]["severe_fragility_condition_count"],
        "material_sensitivity_condition_count": summary[
            "model_summaries"
        ][summary_key]["material_sensitivity_condition_count"],
    }

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "figure_05_input_robustness",
    "generated_at_utc": utc_now(),
    "title": "Input Robustness Under Locked Perturbations",
    "source_files": [
        {
            "source_id": "phase11_all_runs",
            "path": str(all_runs_path),
            "sha256": sha256_file(all_runs_path),
        },
        {
            "source_id": "phase11_summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
    ],
    "evaluation_count": 27,
    "model_count": 3,
    "condition_count": 9,
    "severe_fragility_evaluation_count": (
        summary["severe_fragility_evaluation_count"]
    ),
    "material_sensitivity_evaluation_count": (
        summary["material_sensitivity_evaluation_count"]
    ),
    "model_summaries": model_summary_export,
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
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "known_summary_checks": known_summary_checks,
    "model_inference_performed": False,
    "model_retraining_performed": False,
    "test_used_for_model_selection": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "final_model_changed": False,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "figure_05_input_robustness",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "evaluation_count": 27,
    "model_count": 3,
    "condition_count": 9,
    "severe_fragility_evaluation_count": 5,
    "material_sensitivity_evaluation_count": 9,
    "model_inference_performed": False,
    "model_retraining_performed": False,
    "test_used_for_model_selection": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "final_model_changed": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 FIGURE 05 - INPUT ROBUSTNESS")
print("=" * 96)
print("Models                           : 3")
print("Conditions                       : 9")
print("Locked evaluations               : 27")
print(
    "Severe fragility evaluations    : "
    f"{summary['severe_fragility_evaluation_count']}"
)
print(
    "Material sensitivity evaluations: "
    f"{summary['material_sensitivity_evaluation_count']}"
)
print()

for model_id in MODEL_ORDER:
    model_summary = model_summary_export[model_id]

    print(
        f"{MODEL_LABELS[model_id]:<22} "
        f"clean={model_summary['clean_fingerprint_macro_f1']:.9f} | "
        f"worst={model_summary['worst_condition']} | "
        f"worst_F1={model_summary['worst_fingerprint_macro_f1']:.9f}"
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
print("Model retraining performed       : False")
print("Test used for model selection    : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Final model changed              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 FIGURE 05 GENERATED AND LOCKED")
