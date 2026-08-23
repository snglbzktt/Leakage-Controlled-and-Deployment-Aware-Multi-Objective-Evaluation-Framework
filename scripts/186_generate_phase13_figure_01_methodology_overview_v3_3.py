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

FIGURE_02_LOCK = AUDIT / "phase13_figure_02_lodo_generalization_locked_v3_3.json"
FIGURE_03_LOCK = AUDIT / "phase13_figure_03_compression_tradeoff_locked_v3_3.json"
FIGURE_04_LOCK = AUDIT / "phase13_figure_04_pareto_selection_locked_v3_3.json"
FIGURE_05_LOCK = AUDIT / "phase13_figure_05_input_robustness_locked_v3_3.json"

FIGURE_PNG = FIGURE_DIR / "figure_01_methodology_overview_v3_3.png"
FIGURE_SVG = FIGURE_DIR / "figure_01_methodology_overview_v3_3.svg"
SOURCE_CSV = SOURCE_DIR / "figure_01_methodology_overview_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "figure_01_methodology_overview_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "figure_01_methodology_overview_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_figure_01_methodology_overview_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


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


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
    FIGURE_02_LOCK,
    FIGURE_03_LOCK,
    FIGURE_04_LOCK,
    FIGURE_05_LOCK,
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
            "Figure 01 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)

figure_locks = {
    "figure_02": read_json(FIGURE_02_LOCK),
    "figure_03": read_json(FIGURE_03_LOCK),
    "figure_04": read_json(FIGURE_04_LOCK),
    "figure_05": read_json(FIGURE_05_LOCK),
}

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
    "figures_02_to_05_locked": all(
        lock.get("status") == "locked"
        and lock.get("ready_for_manuscript") is True
        and lock.get("all_checks_passed") is True
        for lock in figure_locks.values()
    ),
    "figure_01_mapped": any(
        asset["asset_id"] == "figure_01_methodology_overview"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Figure 01 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

source_ids = [
    "lodo_global_summary",
    "phase5_closure",
    "phase6_summary",
    "phase8_final_model_lock",
    "phase9_artifact_lock",
    "phase10_release_lock",
    "phase11_lock",
]

source_paths = {
    source_id: get_bound_source(binding, source_id)
    for source_id in source_ids
}

lodo = read_json(source_paths["lodo_global_summary"])
phase5 = read_json(source_paths["phase5_closure"])
phase6 = read_json(source_paths["phase6_summary"])
phase8 = read_json(source_paths["phase8_final_model_lock"])
phase9 = read_json(source_paths["phase9_artifact_lock"])
phase10 = read_json(source_paths["phase10_release_lock"])
phase11 = read_json(source_paths["phase11_lock"])

data_checks = {
    "lodo_9_devices": lodo["scope"]["device_count"] == 9,
    "lodo_3_models": lodo["scope"]["model_count"] == 3,
    "lodo_27_runs": lodo["scope"]["scientific_run_count"] == 27,
    "lodo_gate_passed": lodo["gate"]["passed"] is True,
    "phase5_110_runs": phase5["final_run_count"] == 110,
    "phase5_22_groups": phase5["group_count"] == 22,
    "phase6_110_runs": phase6["run_count"] == 110,
    "phase6_no_val_test_access": (
        phase6["data_access"]["validation_data_access"] is False
        and phase6["data_access"]["test_data_access"] is False
    ),
    "phase8_family_selected": phase8["final_model_family_selected"] is True,
    "phase8_selected_p50_qat": (
        phase8["selected_model_family"]["candidate_id"]
        == "tinyml_mlp::P50-QAT"
    ),
    "phase9_seed_2026": phase9["canonical_seed"] == 2026,
    "phase9_roundtrip_exact": phase9["serialization_roundtrip_exact"] is True,
    "phase10_release_locked": phase10["release_locked"] is True,
    "phase10_deterministic_exact": phase10["deterministic_rebuild_exact"] is True,
    "phase11_27_evaluations": phase11["locked_evaluation_count"] == 27,
    "phase11_locked": phase11["phase11_locked"] is True,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Figure 01 source validation failed: "
        + ", ".join(failed_data_checks)
    )

stages = [
    {
        "order": 1,
        "stage": "Strict LODO generalization",
        "phase": "Phase 3",
        "detail": "9 devices × 3 models = 27 locked runs",
        "evidence": "Global severe-fragility gate passed",
    },
    {
        "order": 2,
        "stage": "Compression matrix",
        "phase": "Phase 5",
        "detail": "2 architectures × 11 variants × 5 seeds = 110 runs",
        "evidence": "22 grouped configurations",
    },
    {
        "order": 3,
        "stage": "Deployment benchmark",
        "phase": "Phase 6",
        "detail": "110 isolated CPU measurements",
        "evidence": "Batch-1 latency + batch-32 throughput; no val/test access",
    },
    {
        "order": 4,
        "stage": "Multi-objective selection",
        "phase": "Phase 8",
        "detail": "Eligibility → Pareto → deployment ordering",
        "evidence": "Selected family: TinyML-MLP P50-QAT",
    },
    {
        "order": 5,
        "stage": "Final deployment artifact",
        "phase": "Phase 9",
        "detail": "Canonical seed 2026",
        "evidence": "Exact source/bundle and serialization round-trip",
    },
    {
        "order": 6,
        "stage": "Deterministic release",
        "phase": "Phase 10",
        "detail": "Locked release archive",
        "evidence": "Deterministic rebuild exact",
    },
    {
        "order": 7,
        "stage": "Input robustness",
        "phase": "Phase 11",
        "detail": "3 models × 9 perturbation conditions = 27 evaluations",
        "evidence": "No retraining, no model reselection, final model unchanged",
    },
]

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "order",
            "phase",
            "stage",
            "detail",
            "evidence",
        ],
    )
    writer.writeheader()
    writer.writerows(stages)

plt.rcParams.update(
    {
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

fig, ax = plt.subplots(figsize=(7.4, 8.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

y_positions = [
    0.91,
    0.78,
    0.65,
    0.52,
    0.39,
    0.26,
    0.13,
]

for index, (stage, y) in enumerate(zip(stages, y_positions)):
    box_text = (
        f"{stage['phase']} — {stage['stage']}\n"
        f"{stage['detail']}\n"
        f"{stage['evidence']}"
    )

    ax.text(
        0.5,
        y,
        box_text,
        ha="center",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.55",
            "fill": False,
            "linewidth": 1.2,
        },
    )

    if index < len(stages) - 1:
        next_y = y_positions[index + 1]

        ax.annotate(
            "",
            xy=(0.5, next_y + 0.052),
            xytext=(0.5, y - 0.052),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.1,
            },
        )

ax.set_title(
    "Locked Experimental and Deployment Workflow",
    fontsize=12,
    pad=14,
)

fig.text(
    0.5,
    0.03,
    "Phase 12 external-dataset validation remains deferred/optional and is not used for the locked primary conclusions.",
    ha="center",
    va="bottom",
    fontsize=7.5,
)

fig.tight_layout(rect=(0.04, 0.055, 0.96, 0.98))
fig.savefig(FIGURE_PNG, bbox_inches="tight")
fig.savefig(FIGURE_SVG, bbox_inches="tight")
plt.close(fig)

caption = (
    "Figure 1. Locked experimental and deployment workflow used for the primary "
    "N-BaIoT study. Strict leave-one-device-out generalization was completed before "
    "the compression study, followed by isolated CPU deployment measurements and "
    "predeclared multi-objective selection. The selected TinyML-MLP P50-QAT family "
    "was then instantiated as a single seed-2026 deployment artifact, packaged into "
    "a deterministic release archive, and evaluated under locked input perturbations. "
    "The robustness stage did not retrain models, reuse test results for selection, "
    "or change the final model. External-dataset validation is deferred/optional."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "figure_01_methodology_overview",
    "generated_at_utc": utc_now(),
    "title": "Locked Experimental and Deployment Workflow",
    "stage_count": len(stages),
    "stages": stages,
    "phase12_external_dataset": "deferred_optional",
    "source_files": [
        {
            "source_id": source_id,
            "path": str(source_paths[source_id]),
            "sha256": sha256_file(source_paths[source_id]),
        }
        for source_id in source_ids
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
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "figure_01_methodology_overview",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "stage_count": 7,
    "phase12_external_dataset": "deferred_optional",
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 FIGURE 01 - METHODOLOGY OVERVIEW")
print("=" * 96)
print("Workflow stages                  : 7")
print("LODO scientific runs             : 27")
print("Compression runs                 : 110")
print("Deployment benchmark runs        : 110")
print("Selected model family            : tinyml_mlp::P50-QAT")
print("Canonical deployment seed        : 2026")
print("Robustness evaluations           : 27")
print("Phase 12 external dataset        : deferred_optional")
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
print("PHASE 13 FIGURE 01 GENERATED AND LOCKED")
