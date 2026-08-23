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
import numpy as np


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

FIGURE_PNG = FIGURE_DIR / "figure_02_lodo_generalization_v3_3.png"
FIGURE_SVG = FIGURE_DIR / "figure_02_lodo_generalization_v3_3.svg"
SOURCE_CSV = SOURCE_DIR / "figure_02_lodo_generalization_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "figure_02_lodo_generalization_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "figure_02_lodo_generalization_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_figure_02_lodo_generalization_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

EXPECTED_MODELS = [
    "compact_dnn_b0",
    "tinyml_mlp_b0",
    "hist_gradient_boosting_b0",
]

MODEL_LABELS = {
    "compact_dnn_b0": "Compact-DNN",
    "tinyml_mlp_b0": "TinyML-MLP",
    "hist_gradient_boosting_b0": "HGB",
}

DEVICE_LABELS = {
    "Danmini_Doorbell": "Danmini Doorbell",
    "Ecobee_Thermostat": "Ecobee Thermostat",
    "Ennio_Doorbell": "Ennio Doorbell†",
    "Philips_B120N10_Baby_Monitor": "Philips Baby Monitor",
    "Provision_PT_737E_Security_Camera": "Provision PT-737E",
    "Provision_PT_838_Security_Camera": "Provision PT-838",
    "Samsung_SNH_1011_N_Webcam": "Samsung Webcam†",
    "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleHome 1002",
    "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleHome 1003",
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
            "Figure 02 output already exists; refusing to overwrite: "
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
    "figure_02_mapped": any(
        asset["asset_id"] == "figure_02_lodo_generalization"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Figure 02 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

matrix_path = get_bound_source(binding, "lodo_device_model_matrix")
summary_path = get_bound_source(binding, "lodo_global_summary")
summary = read_json(summary_path)

rows: list[dict[str, Any]] = []

with matrix_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    reader = csv.DictReader(handle)

    for raw in reader:
        rows.append(
            {
                "held_out_device": raw["held_out_device"],
                "model_id": raw["model_id"],
                "test_fingerprint_macro_f1": float(
                    raw["test_fingerprint_macro_f1"]
                ),
                "test_fingerprint_accuracy": float(
                    raw["test_fingerprint_accuracy"]
                ),
                "test_record_weighted_macro_f1": float(
                    raw["test_record_weighted_macro_f1"]
                ),
                "benign_fnr": float(raw["benign_fnr"]),
                "gafgyt_fnr": float(raw["gafgyt_fnr"]),
                "mirai_fnr": (
                    float(raw["mirai_fnr"])
                    if raw["mirai_fnr"].strip()
                    else None
                ),
                "maximum_eligible_attack_fnr": float(
                    raw["maximum_eligible_attack_fnr"]
                ),
                "local_severe_signal": parse_bool(
                    raw["local_severe_signal"]
                ),
            }
        )

devices = list(summary["scope"]["expected_devices"])
models = list(summary["scope"]["expected_models"])

data_checks = {
    "row_count_27": len(rows) == 27,
    "device_count_9": len(devices) == 9,
    "model_count_3": len(models) == 3,
    "expected_model_set": models == EXPECTED_MODELS,
    "unique_device_model_cells": (
        len(
            {
                (row["held_out_device"], row["model_id"])
                for row in rows
            }
        )
        == 27
    ),
    "no_severe_cells": not any(
        row["local_severe_signal"] for row in rows
    ),
    "global_gate_passed": summary["gate"]["passed"] is True,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Figure 02 source validation failed: "
        + ", ".join(failed_data_checks)
    )

lookup = {
    (row["held_out_device"], row["model_id"]): row
    for row in rows
}

matrix = np.array(
    [
        [
            lookup[(device, model)]["test_fingerprint_macro_f1"]
            for model in models
        ]
        for device in devices
    ],
    dtype=float,
)

device_means = matrix.mean(axis=1, keepdims=True)
plot_matrix = np.concatenate([matrix, device_means], axis=1)

computed_model_means = {
    model: float(matrix[:, index].mean())
    for index, model in enumerate(models)
}

reported_model_means = {
    record["model_id"]: float(record["mean_test_macro_f1"])
    for record in summary["model_ranking"]
}

mean_checks = {
    model: abs(
        computed_model_means[model]
        - reported_model_means[model]
    ) <= 1e-12
    for model in models
}

if not all(mean_checks.values()):
    raise RuntimeError(
        "Computed model means do not match locked global summary: "
        + json.dumps(mean_checks)
    )

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

source_fieldnames = [
    "held_out_device",
    "device_display_name",
    "model_id",
    "model_display_name",
    "test_fingerprint_macro_f1",
    "test_fingerprint_accuracy",
    "test_record_weighted_macro_f1",
    "benign_fnr",
    "gafgyt_fnr",
    "mirai_fnr",
    "maximum_eligible_attack_fnr",
    "local_severe_signal",
]

with SOURCE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(handle, fieldnames=source_fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(
            {
                **row,
                "device_display_name": DEVICE_LABELS[
                    row["held_out_device"]
                ],
                "model_display_name": MODEL_LABELS[
                    row["model_id"]
                ],
            }
        )

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

fig, ax = plt.subplots(figsize=(7.2, 5.6))

image = ax.imshow(
    plot_matrix,
    aspect="auto",
    vmin=0.90,
    vmax=1.00,
)

x_labels = [MODEL_LABELS[model] for model in models] + ["Device mean"]
y_labels = [DEVICE_LABELS[device] for device in devices]

ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=18, ha="right")
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels)

ax.set_title("Leave-One-Device-Out Generalization on N-BaIoT")
ax.set_xlabel("Model")
ax.set_ylabel("Held-out IoT device")

for row_index in range(plot_matrix.shape[0]):
    for column_index in range(plot_matrix.shape[1]):
        value = plot_matrix[row_index, column_index]
        ax.text(
            column_index,
            row_index,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=7.5,
        )

colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
colorbar.set_label("Test fingerprint Macro-F1")

ax.set_xticks(
    np.arange(-0.5, plot_matrix.shape[1], 1),
    minor=True,
)
ax.set_yticks(
    np.arange(-0.5, plot_matrix.shape[0], 1),
    minor=True,
)
ax.grid(which="minor", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

fig.text(
    0.01,
    0.015,
    "† Held-out test set contains two classes; Macro-F1 is computed over classes present. "
    "No device–model cell triggered the locked severe-fragility gate.",
    ha="left",
    va="bottom",
    fontsize=7.5,
)

fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(FIGURE_PNG, bbox_inches="tight")
fig.savefig(FIGURE_SVG, bbox_inches="tight")
plt.close(fig)

caption = (
    "Figure 2. Leave-one-device-out (LODO) generalization across nine N-BaIoT "
    "devices and three model families. Each cell reports test fingerprint Macro-F1 "
    "for one held-out device–model combination; the final column gives the mean "
    "across the three models for each device. Compact-DNN and TinyML-MLP achieved "
    f"mean device-level Macro-F1 values of "
    f"{reported_model_means['compact_dnn_b0']:.4f} and "
    f"{reported_model_means['tinyml_mlp_b0']:.4f}, respectively, while HGB achieved "
    f"{reported_model_means['hist_gradient_boosting_b0']:.4f}. "
    "Ennio Doorbell and Samsung Webcam contain two classes in their held-out test "
    "sets, so Macro-F1 is computed over the classes present. No device–model cell "
    "triggered the locked severe-fragility rule."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "figure_02_lodo_generalization",
    "generated_at_utc": utc_now(),
    "title": "Leave-One-Device-Out Generalization",
    "source_files": [
        {
            "source_id": "lodo_device_model_matrix",
            "path": str(matrix_path),
            "sha256": sha256_file(matrix_path),
        },
        {
            "source_id": "lodo_global_summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
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
    "source_row_count": len(rows),
    "device_count": len(devices),
    "model_count": len(models),
    "computed_model_means": computed_model_means,
    "reported_model_means": reported_model_means,
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "mean_checks": mean_checks,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "figure_02_lodo_generalization",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "source_row_count": 27,
    "device_count": 9,
    "model_count": 3,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 FIGURE 02 - LODO GENERALIZATION")
print("=" * 96)
print(f"Source rows                     : {len(rows)}")
print(f"Held-out devices                : {len(devices)}")
print(f"Models                          : {len(models)}")
print("Severe LODO cells               : 0")
print()
print("Locked model means:")
for model in models:
    print(
        f"  {MODEL_LABELS[model]:<14} : "
        f"{reported_model_means[model]:.9f}"
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
print("PHASE 13 FIGURE 02 GENERATED AND LOCKED")
