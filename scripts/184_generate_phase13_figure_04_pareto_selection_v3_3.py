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

FIGURE_PNG = FIGURE_DIR / "figure_04_pareto_selection_v3_3.png"
FIGURE_SVG = FIGURE_DIR / "figure_04_pareto_selection_v3_3.svg"
SOURCE_CSV = SOURCE_DIR / "figure_04_pareto_selection_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "figure_04_pareto_selection_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "figure_04_pareto_selection_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_figure_04_pareto_selection_locked_v3_3.json"

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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
            "Figure 04 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
figure_02_lock = read_json(FIGURE_02_LOCK)
figure_03_lock = read_json(FIGURE_03_LOCK)

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
    "figure_02_locked": (
        figure_02_lock.get("status") == "locked"
        and figure_02_lock.get("ready_for_manuscript") is True
        and figure_02_lock.get("all_checks_passed") is True
    ),
    "figure_03_locked": (
        figure_03_lock.get("status") == "locked"
        and figure_03_lock.get("ready_for_manuscript") is True
        and figure_03_lock.get("all_checks_passed") is True
    ),
    "figure_04_mapped": any(
        asset["asset_id"] == "figure_04_pareto_selection"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Figure 04 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

input_path = get_bound_source(binding, "phase8_input_matrix")
pareto_path = get_bound_source(binding, "phase8_pareto_results")
trace_path = get_bound_source(binding, "phase8_selection_trace")
final_lock_path = get_bound_source(binding, "phase8_final_model_lock")

input_rows = read_csv_rows(input_path)
pareto_rows = read_csv_rows(pareto_path)
trace_rows = read_csv_rows(trace_path)
final_model_lock = read_json(final_lock_path)

trace_by_profile = {
    row["profile"]: row
    for row in trace_rows
}

expected_profiles = {"strict", "balanced", "relaxed"}

data_checks = {
    "input_rows_22": len(input_rows) == 22,
    "pareto_rows_54": len(pareto_rows) == 54,
    "trace_rows_3": len(trace_rows) == 3,
    "profiles_exact": set(trace_by_profile) == expected_profiles,
    "primary_profile_balanced": final_model_lock["primary_profile"] == "balanced",
    "final_family_selected": final_model_lock["final_model_family_selected"] is True,
    "selected_family_p50_qat": (
        final_model_lock["selected_model_family"]["candidate_id"]
        == "tinyml_mlp::P50-QAT"
    ),
    "weighted_score_not_used": final_model_lock["weighted_score_used"] is False,
    "pairwise_p_not_gate": final_model_lock["pairwise_p_value_used_as_gate"] is False,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Figure 04 source validation failed: "
        + ", ".join(failed_data_checks)
    )

input_lookup = {
    (row["architecture"], row["variant"]): row
    for row in input_rows
}

balanced_rows_raw = [
    row
    for row in pareto_rows
    if row["profile"] == "balanced"
]

balanced_rows: list[dict[str, Any]] = []

for raw in balanced_rows_raw:
    key = (
        raw["architecture"],
        raw["variant"],
    )

    if key not in input_lookup:
        raise RuntimeError(
            "Balanced Pareto candidate not found in input matrix: "
            f"{key}"
        )

    source = input_lookup[key]

    macro_f1 = float(raw["mean_test_fingerprint_macro_f1"])
    state_bytes = float(raw["mean_serialized_state_dict_bytes"])
    latency_ms = float(raw["mean_batch_1_median_ms"])
    throughput = float(raw["mean_batch_32_samples_per_second"])

    consistency_checks = {
        "macro_f1": abs(
            macro_f1
            - float(source["mean_test_fingerprint_macro_f1"])
        ) <= 1e-12,
        "state_bytes": abs(
            state_bytes
            - float(source["mean_serialized_state_dict_bytes"])
        ) <= 1e-9,
        "latency": abs(
            latency_ms
            - float(source["mean_batch_1_median_ms"])
        ) <= 1e-12,
        "throughput": abs(
            throughput
            - float(source["mean_batch_32_samples_per_second"])
        ) <= 1e-9,
    }

    if not all(consistency_checks.values()):
        raise RuntimeError(
            "Phase 8 Pareto/input consistency check failed for "
            f"{key}: {consistency_checks}"
        )

    candidate_id = f"{raw['architecture']}::{raw['variant']}"

    balanced_rows.append(
        {
            "profile": raw["profile"],
            "primary_profile": parse_bool(raw["primary_profile"]),
            "architecture": raw["architecture"],
            "variant": raw["variant"],
            "candidate_id": candidate_id,
            "representation": raw["representation"],
            "eligible": parse_bool(raw["eligible"]),
            "pareto_nondominated": parse_bool(raw["Pareto_nondominated"]),
            "dominated_by_count": int(raw["dominated_by_count"]),
            "deployment_improvement_count": int(
                raw["deployment_improvement_count"]
            ),
            "failed_gates": raw["failed_gates"],
            "mean_test_fingerprint_macro_f1": macro_f1,
            "mean_serialized_state_dict_bytes": state_bytes,
            "mean_state_size_kib": state_bytes / 1024.0,
            "mean_batch_1_median_ms": latency_ms,
            "mean_batch_32_samples_per_second": throughput,
        }
    )

balanced_trace = trace_by_profile["balanced"]
strict_trace = trace_by_profile["strict"]
relaxed_trace = trace_by_profile["relaxed"]

selected_candidate_id = balanced_trace["selected_candidate_id"]
strict_candidate_id = strict_trace["selected_candidate_id"]
relaxed_candidate_id = relaxed_trace["selected_candidate_id"]

selection_checks = {
    "balanced_candidate_count_18": len(balanced_rows) == 18,
    "balanced_eligible_count_11": (
        sum(row["eligible"] for row in balanced_rows) == 11
        and int(balanced_trace["eligible_candidate_count"]) == 11
    ),
    "balanced_pareto_count_5": (
        sum(
            row["eligible"] and row["pareto_nondominated"]
            for row in balanced_rows
        ) == 5
        and int(balanced_trace["Pareto_candidate_count"]) == 5
    ),
    "balanced_selected_p50_qat": (
        selected_candidate_id == "tinyml_mlp::P50-QAT"
    ),
    "strict_selected_p50_fp32_ft": (
        strict_candidate_id == "tinyml_mlp::P50-FP32-FT"
    ),
    "relaxed_selected_p50_qat": (
        relaxed_candidate_id == "tinyml_mlp::P50-QAT"
    ),
    "final_lock_matches_balanced": (
        final_model_lock["selected_model_family"]["candidate_id"]
        == selected_candidate_id
    ),
}

failed_selection_checks = [
    name for name, passed in selection_checks.items() if not passed
]

if failed_selection_checks:
    raise RuntimeError(
        "Figure 04 selection checks failed: "
        + ", ".join(failed_selection_checks)
    )

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

source_fieldnames = list(balanced_rows[0].keys())

with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=source_fieldnames)
    writer.writeheader()
    writer.writerows(balanced_rows)

ineligible_rows = [
    row
    for row in balanced_rows
    if not row["eligible"]
]

eligible_dominated_rows = [
    row
    for row in balanced_rows
    if row["eligible"] and not row["pareto_nondominated"]
]

pareto_rows_balanced = [
    row
    for row in balanced_rows
    if (
        row["eligible"]
        and row["pareto_nondominated"]
        and row["candidate_id"] != selected_candidate_id
    )
]

selected_rows = [
    row
    for row in balanced_rows
    if row["candidate_id"] == selected_candidate_id
]

if len(selected_rows) != 1:
    raise RuntimeError(
        "Expected exactly one balanced selected candidate, found "
        f"{len(selected_rows)}"
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

fig, ax = plt.subplots(figsize=(7.2, 5.2))


def scatter_group(rows: list[dict[str, Any]], marker: str, label: str, size: float) -> None:
    if not rows:
        return

    ax.scatter(
        [row["mean_state_size_kib"] for row in rows],
        [row["mean_test_fingerprint_macro_f1"] for row in rows],
        marker=marker,
        s=size,
        label=label,
    )


scatter_group(
    ineligible_rows,
    "x",
    "Ineligible",
    45,
)

scatter_group(
    eligible_dominated_rows,
    "o",
    "Eligible, dominated",
    45,
)

scatter_group(
    pareto_rows_balanced,
    "D",
    "Pareto non-dominated",
    55,
)

scatter_group(
    selected_rows,
    "*",
    "Balanced-profile selection",
    130,
)

annotated_candidate_ids = {
    selected_candidate_id,
    strict_candidate_id,
}

for row in balanced_rows:
    if row["candidate_id"] in annotated_candidate_ids:
        label = row["candidate_id"].replace("tinyml_mlp::", "TinyML-MLP ")
        ax.annotate(
            label,
            (
                row["mean_state_size_kib"],
                row["mean_test_fingerprint_macro_f1"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
        )

ax.set_xscale("log")
ax.set_xlabel("Mean serialized state size (KiB, log scale)")
ax.set_ylabel("Mean test fingerprint Macro-F1")
ax.set_title("Balanced-Profile Pareto Selection")
ax.grid(True, linewidth=0.5, alpha=0.35)
ax.legend(frameon=True)

fig.text(
    0.01,
    0.015,
    "Balanced profile: 18 compressed candidates, 11 eligible, 5 Pareto non-dominated. "
    "The locked balanced/relaxed selection is TinyML-MLP P50-QAT; "
    "the strict-profile sensitivity choice is TinyML-MLP P50-FP32-FT.",
    ha="left",
    va="bottom",
    fontsize=7.5,
)

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURE_PNG, bbox_inches="tight")
fig.savefig(FIGURE_SVG, bbox_inches="tight")
plt.close(fig)

caption = (
    "Figure 4. Multi-objective candidate selection under the locked balanced "
    "profile. The plot shows 18 compressed candidates in the accuracy–model-size "
    "plane. Eleven candidates passed the balanced eligibility gates and five were "
    "Pareto non-dominated. TinyML-MLP P50-QAT was selected under the balanced and "
    "relaxed profiles, while TinyML-MLP P50-FP32-FT was selected under the strict "
    "sensitivity profile. Selection followed the predeclared eligibility–Pareto–"
    "deployment ordering rule; no weighted score and no pairwise p-value gate were used."
)

atomic_text(CAPTION_TXT, caption + "\n")

selected_row = selected_rows[0]

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "figure_04_pareto_selection",
    "generated_at_utc": utc_now(),
    "title": "Balanced-Profile Pareto Selection",
    "source_files": [
        {
            "source_id": "phase8_input_matrix",
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        {
            "source_id": "phase8_pareto_results",
            "path": str(pareto_path),
            "sha256": sha256_file(pareto_path),
        },
        {
            "source_id": "phase8_selection_trace",
            "path": str(trace_path),
            "sha256": sha256_file(trace_path),
        },
        {
            "source_id": "phase8_final_model_lock",
            "path": str(final_lock_path),
            "sha256": sha256_file(final_lock_path),
        },
    ],
    "balanced_profile": {
        "compressed_candidate_count": len(balanced_rows),
        "eligible_candidate_count": sum(
            row["eligible"] for row in balanced_rows
        ),
        "pareto_candidate_count": sum(
            row["eligible"] and row["pareto_nondominated"]
            for row in balanced_rows
        ),
        "selected_candidate_id": selected_candidate_id,
        "selected_representation": selected_row["representation"],
        "selected_mean_macro_f1": selected_row[
            "mean_test_fingerprint_macro_f1"
        ],
        "selected_mean_state_size_kib": selected_row["mean_state_size_kib"],
        "selected_mean_batch_1_median_ms": selected_row[
            "mean_batch_1_median_ms"
        ],
        "selected_mean_batch_32_samples_per_second": selected_row[
            "mean_batch_32_samples_per_second"
        ],
    },
    "sensitivity_profile_selections": {
        "strict": strict_candidate_id,
        "balanced": selected_candidate_id,
        "relaxed": relaxed_candidate_id,
    },
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
    "selection_checks": selection_checks,
    "weighted_score_used": False,
    "pairwise_p_value_used_as_gate": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "figure_04_pareto_selection",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "selected_candidate_id": selected_candidate_id,
    "balanced_eligible_candidate_count": 11,
    "balanced_pareto_candidate_count": 5,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 FIGURE 04 - PARETO SELECTION")
print("=" * 96)
print(f"Balanced compressed candidates   : {len(balanced_rows)}")
print(
    "Balanced eligible candidates     : "
    f"{sum(row['eligible'] for row in balanced_rows)}"
)
print(
    "Balanced Pareto candidates       : "
    f"{sum(row['eligible'] and row['pareto_nondominated'] for row in balanced_rows)}"
)
print(f"Balanced selection               : {selected_candidate_id}")
print(f"Strict sensitivity selection     : {strict_candidate_id}")
print(f"Relaxed sensitivity selection    : {relaxed_candidate_id}")
print()
print(
    "Selected mean Macro-F1           : "
    f"{selected_row['mean_test_fingerprint_macro_f1']:.9f}"
)
print(
    "Selected mean state size         : "
    f"{selected_row['mean_state_size_kib']:.3f} KiB"
)
print(
    "Selected batch-1 median latency  : "
    f"{selected_row['mean_batch_1_median_ms']:.6f} ms"
)
print(
    "Selected batch-32 throughput     : "
    f"{selected_row['mean_batch_32_samples_per_second']:.3f} samples/s"
)
print()
print(f"PNG                              : {FIGURE_PNG}")
print(f"SVG                              : {FIGURE_SVG}")
print(f"Source CSV                       : {SOURCE_CSV}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Weighted score used              : False")
print("Pairwise p-value used as gate    : False")
print("Model inference performed        : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 FIGURE 04 GENERATED AND LOCKED")
