from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
PROTOCOLS = ROOT / "configs" / "protocols"

PROTOCOL_VERSION = "phase8_validation_only_selection_v3_3"
SOURCE_PHASE8_VERSION = "phase8_multi_objective_decision_v3_2"
PRIMARY_PROFILE = "balanced"
NUMERIC_TOLERANCE = 1e-12

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
    "DQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
)

COMPRESSED_CANDIDATES = (
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
    "DQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

# Numeric tolerances are inherited unchanged from the original Phase 8
# profiles. The original raw-weighted TEST gate and TEST-derived
# fragility gate are deliberately not carried into this corrective
# validation-only analysis because a uniform validation counterpart is
# not available across every variant.
GATE_PROFILES = {
    "strict": {
        "maximum_mean_validation_macro_f1_loss": 0.0010,
        "maximum_worst_seed_validation_macro_f1_loss": 0.0025,
    },
    "balanced": {
        "maximum_mean_validation_macro_f1_loss": 0.0020,
        "maximum_worst_seed_validation_macro_f1_loss": 0.0050,
    },
    "relaxed": {
        "maximum_mean_validation_macro_f1_loss": 0.0050,
        "maximum_worst_seed_validation_macro_f1_loss": 0.0100,
    },
}

PARETO_OBJECTIVES = (
    {
        "metric": "mean_validation_fingerprint_macro_f1",
        "direction": "maximize",
        "role": "predictive_quality",
    },
    {
        "metric": "mean_serialized_state_dict_bytes",
        "direction": "minimize",
        "role": "storage",
    },
    {
        "metric": "mean_batch_1_median_ms",
        "direction": "minimize",
        "role": "single_sample_latency",
    },
    {
        "metric": "mean_batch_32_samples_per_second",
        "direction": "maximize",
        "role": "throughput",
    },
)

PRIMARY_SELECTION_ORDER = (
    {
        "criterion": "eligible_under_balanced_profile",
        "direction": "required_true",
    },
    {
        "criterion": "Pareto_nondominated",
        "direction": "required_true",
    },
    {
        "criterion": "mean_serialized_state_dict_bytes",
        "direction": "ascending",
    },
    {
        "criterion": "mean_batch_1_median_ms",
        "direction": "ascending",
    },
    {
        "criterion": "mean_batch_32_samples_per_second",
        "direction": "descending",
    },
    {
        "criterion": "mean_validation_fingerprint_macro_f1",
        "direction": "descending",
    },
    {
        "criterion": "architecture",
        "direction": "lexicographic_ascending",
    },
    {
        "criterion": "variant",
        "direction": "lexicographic_ascending",
    },
)

# Explicit Phase-5 validation sources. Only validation fields listed
# below are extracted. No Phase-7 source and no original Phase-8 input
# matrix is read by this script.
TINYML_B0 = AUDIT / "phase5_tinyml_mlp_b0_runs_v3_2.csv"
COMPACT_B0 = AUDIT / "phase5_compact_dnn_b0_runs_v3_2.csv"
FP32_FT = AUDIT / "phase5_FP32_FT_all_runs_v3_2.csv"
PRUNING_NOFT = AUDIT / "phase5_pruning_noFT_evaluation_all_runs_v3_2.csv"
PRUNING_FP32_FT = AUDIT / "phase5_pruning_FP32_FT_all_runs_v3_2.csv"
DQ_RUNS = AUDIT / "phase5_DQ_all_runs_v3_2.csv"
QAT_RUNS = AUDIT / "phase5_QAT_all_runs_v3_2.csv"
PTQ_SELECTION_LOCK = AUDIT / "phase5_PTQ_validation_selection_locked_v3_2.json"
PTQ_VERIFIED_CANDIDATES = AUDIT / "phase5_PTQ_validation_sweep_verified_candidates_v3_2.csv"

PHASE6_GROUP = AUDIT / "phase6_deployment_benchmark_group_summary_v3_2.csv"
PHASE6_LOCK = AUDIT / "phase6_deployment_benchmark_locked_v3_2.json"

OUTPUT_PROTOCOL = PROTOCOLS / "phase8_validation_only_selection_protocol_v3_3.json"
OUTPUT_VALIDATION_SOURCE = AUDIT / "phase8_validation_only_source_matrix_v3_3.csv"
OUTPUT_DECISION = AUDIT / "phase8_validation_only_decision_matrix_v3_3.csv"
OUTPUT_ELIGIBILITY = AUDIT / "phase8_validation_only_eligibility_v3_3.csv"
OUTPUT_PARETO = AUDIT / "phase8_validation_only_pareto_results_v3_3.csv"
OUTPUT_TRACE = AUDIT / "phase8_validation_only_selection_trace_v3_3.csv"
OUTPUT_SUMMARY = AUDIT / "phase8_validation_only_selection_summary_v3_3.json"
OUTPUT_LOCK = AUDIT / "phase8_validation_only_selection_locked_v3_3.json"
OUTPUT_MANIFEST = AUDIT / "phase8_validation_only_selection_lock_manifest_v3_3.json"

OUTPUTS = (
    OUTPUT_PROTOCOL,
    OUTPUT_VALIDATION_SOURCE,
    OUTPUT_DECISION,
    OUTPUT_ELIGIBILITY,
    OUTPUT_PARETO,
    OUTPUT_TRACE,
    OUTPUT_SUMMARY,
    OUTPUT_LOCK,
    OUTPUT_MANIFEST,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"Non-finite value in {field_name}: {value}"
        )
    return number


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Cannot parse boolean: {value}")


def read_selected_columns(
    path: Path,
    required_columns: list[str],
) -> list[dict[str, str]]:
    """
    Extract only explicitly allowed columns from a CSV.

    Some historical Phase-5 CSVs also contain test columns. Those fields
    are neither selected nor propagated by this corrective analysis.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError(f"Empty CSV: {path}") from error

        positions = {name: index for index, name in enumerate(header)}
        missing = [
            name for name in required_columns
            if name not in positions
        ]
        if missing:
            raise RuntimeError(
                f"Missing columns in {path}: {missing}; "
                f"observed={header}"
            )

        indices = [positions[name] for name in required_columns]
        rows: list[dict[str, str]] = []
        for raw in reader:
            if not raw:
                continue
            if len(raw) != len(header):
                raise RuntimeError(
                    f"Malformed row in {path}: "
                    f"expected {len(header)} columns, got {len(raw)}"
                )
            rows.append(
                {
                    name: raw[index]
                    for name, index in zip(
                        required_columns,
                        indices,
                        strict=True,
                    )
                }
            )
        return rows


def normalize_pruning_ft_variant(value: str) -> str:
    mapping = {
        "P25": "P25-FP32-FT",
        "P50": "P50-FP32-FT",
        "P25-FP32-FT": "P25-FP32-FT",
        "P50-FP32-FT": "P50-FP32-FT",
    }
    if value not in mapping:
        raise RuntimeError(
            f"Unexpected pruning FP32-FT variant: {value}"
        )
    return mapping[value]


def append_validation_rows(
    target: list[dict[str, Any]],
    path: Path,
    metric_column: str,
    *,
    fixed_architecture: str | None = None,
    fixed_variant: str | None = None,
    variant_transform: Callable[[str], str] | None = None,
) -> None:
    columns = ["seed", metric_column]
    if fixed_architecture is None:
        columns.insert(0, "architecture")
    if fixed_variant is None:
        insert_at = 1 if fixed_architecture is None else 0
        columns.insert(insert_at, "variant")

    rows = read_selected_columns(path, columns)

    for row in rows:
        architecture = (
            fixed_architecture
            if fixed_architecture is not None
            else row["architecture"]
        )
        variant = (
            fixed_variant
            if fixed_variant is not None
            else row["variant"]
        )
        if variant_transform is not None:
            variant = variant_transform(variant)

        target.append(
            {
                "architecture": architecture,
                "variant": variant,
                "seed": int(row["seed"]),
                "validation_fingerprint_macro_f1": finite_float(
                    row[metric_column],
                    metric_column,
                ),
                "validation_metric_source": metric_column,
                "source_file": str(path),
                "source_file_sha256": sha256_file(path),
            }
        )


def collect_validation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    append_validation_rows(
        rows,
        TINYML_B0,
        "best_validation_macro_f1",
        fixed_architecture="tinyml_mlp",
        fixed_variant="B0",
    )
    append_validation_rows(
        rows,
        COMPACT_B0,
        "best_validation_macro_f1",
        fixed_architecture="compact_dnn",
        fixed_variant="B0",
    )
    append_validation_rows(
        rows,
        FP32_FT,
        "best_validation_macro_f1",
        fixed_variant="FP32-FT",
    )
    append_validation_rows(
        rows,
        PRUNING_NOFT,
        "validation_fingerprint_macro_f1",
    )
    append_validation_rows(
        rows,
        PRUNING_FP32_FT,
        "best_validation_macro_f1",
        variant_transform=normalize_pruning_ft_variant,
    )
    append_validation_rows(
        rows,
        DQ_RUNS,
        "validation_fingerprint_macro_f1",
        fixed_variant="DQ",
    )
    append_validation_rows(
        rows,
        QAT_RUNS,
        "best_validation_INT8_macro_f1",
    )

    # PTQ is special: the calibration size was already selected using a
    # locked validation-only sweep with test_access_count=0. Reuse exactly
    # the verified candidates from that selected size.
    ptq_lock = read_json(PTQ_SELECTION_LOCK)
    selected_sizes = {
        key: int(value)
        for key, value in ptq_lock[
            "selected_calibration_sizes"
        ].items()
    }

    if ptq_lock.get("test_access_count") != 0:
        raise RuntimeError(
            "PTQ validation-selection lock does not report test_access_count=0."
        )

    ptq_rows = read_selected_columns(
        PTQ_VERIFIED_CANDIDATES,
        [
            "architecture",
            "seed",
            "calibration_size",
            "validation_fingerprint_macro_f1",
            "test_access_count",
            "all_checks_passed",
        ],
    )

    selected_ptq_count = 0
    for row in ptq_rows:
        architecture = row["architecture"]
        seed = int(row["seed"])
        calibration_size = int(row["calibration_size"])

        if architecture not in selected_sizes:
            continue
        if calibration_size != selected_sizes[architecture]:
            continue

        if int(row["test_access_count"]) != 0:
            raise RuntimeError(
                "Selected PTQ candidate has non-zero test access."
            )
        if not parse_bool(row["all_checks_passed"]):
            raise RuntimeError(
                "Selected PTQ candidate failed verification."
            )

        rows.append(
            {
                "architecture": architecture,
                "variant": "PTQ",
                "seed": seed,
                "validation_fingerprint_macro_f1": finite_float(
                    row["validation_fingerprint_macro_f1"],
                    "validation_fingerprint_macro_f1",
                ),
                "validation_metric_source": (
                    "PTQ selected calibration / "
                    "validation_fingerprint_macro_f1"
                ),
                "source_file": str(PTQ_VERIFIED_CANDIDATES),
                "source_file_sha256": sha256_file(
                    PTQ_VERIFIED_CANDIDATES
                ),
            }
        )
        selected_ptq_count += 1

    if selected_ptq_count != 10:
        raise RuntimeError(
            "Expected 10 selected PTQ validation rows, "
            f"observed={selected_ptq_count}"
        )

    rows.sort(
        key=lambda row: (
            ARCHITECTURES.index(row["architecture"]),
            VARIANTS.index(row["variant"]),
            SEEDS.index(int(row["seed"])),
        )
    )

    expected_keys = {
        (architecture, variant, seed)
        for architecture in ARCHITECTURES
        for variant in VARIANTS
        for seed in SEEDS
    }
    observed_keys = {
        (
            row["architecture"],
            row["variant"],
            int(row["seed"]),
        )
        for row in rows
    }

    if len(rows) != 110:
        raise RuntimeError(
            f"Validation source row count mismatch: {len(rows)} != 110"
        )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            "Validation source key mismatch. "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    if len(observed_keys) != len(rows):
        raise RuntimeError(
            "Duplicate architecture/variant/seed validation rows detected."
        )

    return rows


def collect_deployment_rows() -> dict[tuple[str, str], dict[str, Any]]:
    phase6_lock = read_json(PHASE6_LOCK)

    if (
        phase6_lock.get("status") != "locked"
        or phase6_lock.get("all_checks_passed") is not True
    ):
        raise RuntimeError(
            "Phase 6 deployment benchmark is not locked/passed."
        )

    columns = [
        "architecture",
        "variant",
        "representation",
        "mean_serialized_state_dict_bytes",
        "mean_batch_1_median_ms",
        "mean_batch_32_samples_per_second",
        "mean_peak_inference_RSS_delta_bytes",
        "validation_data_access",
        "test_data_access",
    ]
    raw_rows = read_selected_columns(
        PHASE6_GROUP,
        columns,
    )

    result: dict[tuple[str, str], dict[str, Any]] = {}

    for row in raw_rows:
        if parse_bool(row["validation_data_access"]):
            raise RuntimeError(
                "Phase 6 group reports validation data access."
            )
        if parse_bool(row["test_data_access"]):
            raise RuntimeError(
                "Phase 6 group reports test data access."
            )

        key = (
            row["architecture"],
            row["variant"],
        )
        if key in result:
            raise RuntimeError(
                f"Duplicate Phase 6 deployment group: {key}"
            )

        result[key] = {
            "architecture": row["architecture"],
            "variant": row["variant"],
            "representation": row["representation"],
            "mean_serialized_state_dict_bytes": finite_float(
                row["mean_serialized_state_dict_bytes"],
                "mean_serialized_state_dict_bytes",
            ),
            "mean_batch_1_median_ms": finite_float(
                row["mean_batch_1_median_ms"],
                "mean_batch_1_median_ms",
            ),
            "mean_batch_32_samples_per_second": finite_float(
                row["mean_batch_32_samples_per_second"],
                "mean_batch_32_samples_per_second",
            ),
            "mean_peak_inference_RSS_delta_bytes": finite_float(
                row["mean_peak_inference_RSS_delta_bytes"],
                "mean_peak_inference_RSS_delta_bytes",
            ),
        }

    expected = {
        (architecture, variant)
        for architecture in ARCHITECTURES
        for variant in VARIANTS
    }
    if set(result) != expected:
        raise RuntimeError(
            "Phase 6 deployment group matrix is incomplete."
        )

    return result


def build_decision_rows(
    validation_rows: list[dict[str, Any]],
    deployment: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    run_lookup = {
        (
            row["architecture"],
            row["variant"],
            int(row["seed"]),
        ): row
        for row in validation_rows
    }

    decision_rows: list[dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        baseline_values = [
            run_lookup[
                (
                    architecture,
                    "B0",
                    seed,
                )
            ][
                "validation_fingerprint_macro_f1"
            ]
            for seed in SEEDS
        ]
        baseline_mean = sum(baseline_values) / len(baseline_values)
        baseline_deployment = deployment[
            (
                architecture,
                "B0",
            )
        ]

        for variant in VARIANTS:
            values = [
                run_lookup[
                    (
                        architecture,
                        variant,
                        seed,
                    )
                ][
                    "validation_fingerprint_macro_f1"
                ]
                for seed in SEEDS
            ]
            mean_validation = sum(values) / len(values)
            population_variance = (
                sum(
                    (value - mean_validation) ** 2
                    for value in values
                )
                / len(values)
            )
            std_validation = math.sqrt(population_variance)
            seed_losses = [
                baseline_value - candidate_value
                for baseline_value, candidate_value
                in zip(
                    baseline_values,
                    values,
                    strict=True,
                )
            ]

            dep = deployment[
                (
                    architecture,
                    variant,
                )
            ]

            state_ratio = (
                dep["mean_serialized_state_dict_bytes"]
                / baseline_deployment[
                    "mean_serialized_state_dict_bytes"
                ]
            )
            latency_ratio = (
                dep["mean_batch_1_median_ms"]
                / baseline_deployment[
                    "mean_batch_1_median_ms"
                ]
            )
            throughput_ratio = (
                dep["mean_batch_32_samples_per_second"]
                / baseline_deployment[
                    "mean_batch_32_samples_per_second"
                ]
            )

            role = (
                "baseline_reference"
                if variant == "B0"
                else (
                    "uncompressed_control"
                    if variant == "FP32-FT"
                    else "compressed_candidate"
                )
            )

            decision_rows.append(
                {
                    "architecture": architecture,
                    "variant": variant,
                    "candidate_role": role,
                    "representation": dep["representation"],
                    "mean_validation_fingerprint_macro_f1": mean_validation,
                    "std_validation_fingerprint_macro_f1": std_validation,
                    "mean_validation_macro_f1_loss_vs_architecture_B0": (
                        baseline_mean - mean_validation
                    ),
                    "worst_seed_validation_macro_f1_loss_vs_architecture_B0": (
                        max(seed_losses)
                    ),
                    "minimum_validation_fingerprint_macro_f1": min(values),
                    "maximum_validation_fingerprint_macro_f1": max(values),
                    "mean_serialized_state_dict_bytes": (
                        dep[
                            "mean_serialized_state_dict_bytes"
                        ]
                    ),
                    "state_size_ratio_vs_architecture_B0": state_ratio,
                    "state_size_reduction_fraction_vs_architecture_B0": (
                        1.0 - state_ratio
                    ),
                    "mean_batch_1_median_ms": (
                        dep["mean_batch_1_median_ms"]
                    ),
                    "batch_1_latency_ratio_vs_architecture_B0": (
                        latency_ratio
                    ),
                    "batch_1_latency_reduction_fraction_vs_architecture_B0": (
                        1.0 - latency_ratio
                    ),
                    "mean_batch_32_samples_per_second": (
                        dep[
                            "mean_batch_32_samples_per_second"
                        ]
                    ),
                    "batch_32_throughput_ratio_vs_architecture_B0": (
                        throughput_ratio
                    ),
                    "batch_32_throughput_gain_fraction_vs_architecture_B0": (
                        throughput_ratio - 1.0
                    ),
                    "mean_peak_inference_RSS_delta_bytes": (
                        dep[
                            "mean_peak_inference_RSS_delta_bytes"
                        ]
                    ),
                }
            )

    if len(decision_rows) != 22:
        raise RuntimeError(
            f"Decision matrix row count mismatch: {len(decision_rows)}"
        )

    return decision_rows


def dominates(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    comparisons = (
        (
            left["mean_validation_fingerprint_macro_f1"],
            right["mean_validation_fingerprint_macro_f1"],
            "maximize",
        ),
        (
            left["mean_serialized_state_dict_bytes"],
            right["mean_serialized_state_dict_bytes"],
            "minimize",
        ),
        (
            left["mean_batch_1_median_ms"],
            right["mean_batch_1_median_ms"],
            "minimize",
        ),
        (
            left["mean_batch_32_samples_per_second"],
            right["mean_batch_32_samples_per_second"],
            "maximize",
        ),
    )

    all_not_worse = True
    at_least_one_better = False

    for left_value, right_value, direction in comparisons:
        if direction == "maximize":
            if left_value < right_value - NUMERIC_TOLERANCE:
                all_not_worse = False
                break
            if left_value > right_value + NUMERIC_TOLERANCE:
                at_least_one_better = True
        else:
            if left_value > right_value + NUMERIC_TOLERANCE:
                all_not_worse = False
                break
            if left_value < right_value - NUMERIC_TOLERANCE:
                at_least_one_better = True

    return all_not_worse and at_least_one_better


def selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mean_serialized_state_dict_bytes"],
        row["mean_batch_1_median_ms"],
        -row["mean_batch_32_samples_per_second"],
        -row["mean_validation_fingerprint_macro_f1"],
        row["architecture"],
        row["variant"],
    )


def compute_selection(
    decision_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    eligibility_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for profile, thresholds in GATE_PROFILES.items():
        candidates: list[dict[str, Any]] = []

        for row in decision_rows:
            size_improved = (
                row[
                    "state_size_reduction_fraction_vs_architecture_B0"
                ]
                > NUMERIC_TOLERANCE
            )
            latency_improved = (
                row[
                    "batch_1_latency_reduction_fraction_vs_architecture_B0"
                ]
                > NUMERIC_TOLERANCE
            )
            throughput_improved = (
                row[
                    "batch_32_throughput_gain_fraction_vs_architecture_B0"
                ]
                > NUMERIC_TOLERANCE
            )
            deployment_improvement_count = sum(
                (
                    size_improved,
                    latency_improved,
                    throughput_improved,
                )
            )

            role_gate = (
                row["candidate_role"]
                == "compressed_candidate"
            )
            mean_gate = (
                row[
                    "mean_validation_macro_f1_loss_vs_architecture_B0"
                ]
                <= (
                    thresholds[
                        "maximum_mean_validation_macro_f1_loss"
                    ]
                    + NUMERIC_TOLERANCE
                )
            )
            worst_gate = (
                row[
                    "worst_seed_validation_macro_f1_loss_vs_architecture_B0"
                ]
                <= (
                    thresholds[
                        "maximum_worst_seed_validation_macro_f1_loss"
                    ]
                    + NUMERIC_TOLERANCE
                )
            )
            deployment_gate = (
                deployment_improvement_count >= 1
            )

            eligible = all(
                (
                    role_gate,
                    mean_gate,
                    worst_gate,
                    deployment_gate,
                )
            )

            failed: list[str] = []
            if not role_gate:
                failed.append("candidate_role")
            if not mean_gate:
                failed.append("mean_validation_macro_f1_loss")
            if not worst_gate:
                failed.append(
                    "worst_seed_validation_macro_f1_loss"
                )
            if not deployment_gate:
                failed.append("deployment_improvement")

            eligibility_rows.append(
                {
                    "profile": profile,
                    "primary_profile": (
                        profile == PRIMARY_PROFILE
                    ),
                    "architecture": row["architecture"],
                    "variant": row["variant"],
                    "candidate_role": row["candidate_role"],
                    "representation": row["representation"],
                    "role_gate_passed": role_gate,
                    "mean_validation_macro_f1_gate_passed": mean_gate,
                    "worst_seed_validation_macro_f1_gate_passed": worst_gate,
                    "deployment_gate_passed": deployment_gate,
                    "deployment_improvement_count": (
                        deployment_improvement_count
                    ),
                    "eligible": eligible,
                    "failed_gates": "|".join(failed),
                }
            )

            if eligible:
                candidate = dict(row)
                candidate[
                    "deployment_improvement_count"
                ] = deployment_improvement_count
                candidates.append(candidate)

        pareto_candidates: list[dict[str, Any]] = []

        for row in candidates:
            dominators = [
                other
                for other in candidates
                if (
                    other is not row
                    and dominates(other, row)
                )
            ]
            is_pareto = len(dominators) == 0

            pareto_rows.append(
                {
                    "profile": profile,
                    "primary_profile": (
                        profile == PRIMARY_PROFILE
                    ),
                    "architecture": row["architecture"],
                    "variant": row["variant"],
                    "representation": row["representation"],
                    "eligible": True,
                    "Pareto_nondominated": is_pareto,
                    "dominated_by_count": len(dominators),
                    "dominated_by": "|".join(
                        sorted(
                            f"{item['architecture']}::{item['variant']}"
                            for item in dominators
                        )
                    ),
                    "mean_validation_fingerprint_macro_f1": (
                        row[
                            "mean_validation_fingerprint_macro_f1"
                        ]
                    ),
                    "mean_serialized_state_dict_bytes": (
                        row[
                            "mean_serialized_state_dict_bytes"
                        ]
                    ),
                    "mean_batch_1_median_ms": (
                        row["mean_batch_1_median_ms"]
                    ),
                    "mean_batch_32_samples_per_second": (
                        row[
                            "mean_batch_32_samples_per_second"
                        ]
                    ),
                    "deployment_improvement_count": (
                        row[
                            "deployment_improvement_count"
                        ]
                    ),
                }
            )

            if is_pareto:
                pareto_candidates.append(row)

        selected = (
            sorted(
                pareto_candidates,
                key=selection_sort_key,
            )[0]
            if pareto_candidates
            else None
        )

        trace_rows.append(
            {
                "profile": profile,
                "primary_profile": (
                    profile == PRIMARY_PROFILE
                ),
                "compressed_candidate_count": 18,
                "eligible_candidate_count": len(candidates),
                "Pareto_candidate_count": len(pareto_candidates),
                "selection_available": selected is not None,
                "selected_architecture": (
                    selected["architecture"]
                    if selected is not None
                    else ""
                ),
                "selected_variant": (
                    selected["variant"]
                    if selected is not None
                    else ""
                ),
                "selected_representation": (
                    selected["representation"]
                    if selected is not None
                    else ""
                ),
                "selected_candidate_id": (
                    (
                        f"{selected['architecture']}"
                        f"::{selected['variant']}"
                    )
                    if selected is not None
                    else ""
                ),
                "selected_mean_validation_macro_f1": (
                    selected[
                        "mean_validation_fingerprint_macro_f1"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_mean_state_bytes": (
                    selected[
                        "mean_serialized_state_dict_bytes"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_mean_batch_1_median_ms": (
                    selected[
                        "mean_batch_1_median_ms"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_mean_batch_32_samples_per_second": (
                    selected[
                        "mean_batch_32_samples_per_second"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_mean_validation_macro_f1_loss_vs_B0": (
                    selected[
                        "mean_validation_macro_f1_loss_vs_architecture_B0"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_worst_seed_validation_macro_f1_loss_vs_B0": (
                    selected[
                        "worst_seed_validation_macro_f1_loss_vs_architecture_B0"
                    ]
                    if selected is not None
                    else ""
                ),
                "selected_deployment_improvement_count": (
                    selected[
                        "deployment_improvement_count"
                    ]
                    if selected is not None
                    else ""
                ),
                "selection_rule": (
                    "eligible -> Pareto -> state asc -> "
                    "batch1 latency asc -> throughput desc -> "
                    "validation Macro-F1 desc -> lexical tie-break"
                ),
            }
        )

    return (
        eligibility_rows,
        pareto_rows,
        trace_rows,
    )


def validate_no_test_fields_in_selection_artifacts(
    rows_by_name: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> None:
    """
    Guard against test-derived metrics entering the actual decision logic.

    Audit metadata such as
    ``test_metrics_used_for_selection=False`` is intentionally allowed.
    Only fields that participate in eligibility, Pareto objectives, or
    deterministic ordering are forbidden from referencing ``test_*``.
    """
    offenders: list[str] = []

    for name, rows in rows_by_name.items():
        if not rows:
            continue
        for field in rows[0].keys():
            if field.lower().startswith("test_"):
                offenders.append(f"{name}:{field}")

    for objective in protocol["Pareto_analysis"]["objectives"]:
        metric = str(objective.get("metric", ""))
        if metric.lower().startswith("test_"):
            offenders.append(f"pareto_objective:{metric}")

    for item in protocol["primary_selection_rule"]["order"]:
        criterion = str(item.get("criterion", ""))
        if criterion.lower().startswith("test_"):
            offenders.append(f"selection_order:{criterion}")

    for profile_name, thresholds in protocol["eligibility_profiles"].items():
        for threshold_name in thresholds:
            if str(threshold_name).lower().startswith("test_"):
                offenders.append(
                    f"eligibility_profile:{profile_name}:{threshold_name}"
                )

    if offenders:
        raise RuntimeError(
            "Forbidden test-derived decision field entered selection logic: "
            + ", ".join(offenders)
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Corrective validation-only Phase 8 multi-objective "
            "selection. Preserves original v3.2 Phase 8 artifacts."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite only the new validation-only v3.3 artifacts. "
            "Original v3.2 Phase 8 artifacts are never modified."
        ),
    )
    args = parser.parse_args()

    required = (
        TINYML_B0,
        COMPACT_B0,
        FP32_FT,
        PRUNING_NOFT,
        PRUNING_FP32_FT,
        DQ_RUNS,
        QAT_RUNS,
        PTQ_SELECTION_LOCK,
        PTQ_VERIFIED_CANDIDATES,
        PHASE6_GROUP,
        PHASE6_LOCK,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required corrective-selection sources missing:\n"
            + "\n".join(str(path) for path in missing)
        )

    existing = [path for path in OUTPUTS if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Corrected v3.3 artifacts already exist; refusing to "
            "overwrite. Re-run with --overwrite only after reviewing "
            "them:\n"
            + "\n".join(str(path) for path in existing)
        )

    if args.overwrite:
        for path in OUTPUTS:
            if path.exists():
                path.unlink()

    print("=" * 96)
    print("PHASE 8 VALIDATION-ONLY CORRECTED MULTI-OBJECTIVE SELECTION v3.3")
    print("=" * 96)
    print("Original Phase 8 v3.2 modified  : False")
    print("Phase 7 p-values loaded         : False")
    print("Original Phase 8 input loaded   : False")
    print("Test arrays loaded              : False")
    print("Test metrics used in decision   : False")
    print("Validation predictive metric    : fingerprint Macro-F1")
    print("Deployment source               : Phase 6 locked benchmark")
    print("Primary profile                 : balanced")
    print()

    validation_rows = collect_validation_rows()
    deployment = collect_deployment_rows()
    decision_rows = build_decision_rows(
        validation_rows,
        deployment,
    )

    (
        eligibility_rows,
        pareto_rows,
        trace_rows,
    ) = compute_selection(decision_rows)

    protocol = {
        "status": "corrective_reanalysis_protocol",
        "phase": 8,
        "protocol_version": PROTOCOL_VERSION,
        "supersedes_for_model_family_selection": (
            SOURCE_PHASE8_VERSION
        ),
        "created_at_utc": utc_now(),
        "scientific_role": (
            "post-hoc corrective reanalysis after identifying that "
            "the original Phase 8 family-level decision consumed "
            "test-derived predictive summaries"
        ),
        "primary_profile": PRIMARY_PROFILE,
        "predictive_selection_source": (
            "Phase-5 validation metrics only"
        ),
        "deployment_source": (
            "Phase-6 locked benchmark; validation_data_access=False "
            "and test_data_access=False required"
        ),
        "eligibility_profiles": GATE_PROFILES,
        "eligibility_policy": {
            "required_candidate_role": "compressed_candidate",
            "minimum_deployment_improvement_count": 1,
            "deployment_improvements": [
                "smaller serialized state than architecture B0",
                "lower batch-1 median latency than architecture B0",
                "higher batch-32 throughput than architecture B0",
            ],
            "raw_weighted_gate_removed": True,
            "reason_raw_weighted_gate_removed": (
                "A uniform validation raw-weighted metric is not "
                "available for every compression variant."
            ),
            "test_fragility_gate_removed": True,
            "reason_test_fragility_gate_removed": (
                "The historical fragility flag is derived from test "
                "performance and therefore cannot be part of a "
                "validation-only selection gate."
            ),
        },
        "Pareto_analysis": {
            "objectives": list(PARETO_OBJECTIVES),
            "weighted_score_used": False,
        },
        "primary_selection_rule": {
            "order": list(PRIMARY_SELECTION_ORDER),
            "p_values_used_as_exclusion_gate": False,
            "robustness_results_used_for_selection": False,
            "test_metrics_used_for_selection": False,
        },
        "reporting_policy": {
            "final test metrics": (
                "report only after model-family selection; not an "
                "input to this corrected decision"
            ),
            "original Phase 8 artifacts": (
                "retained unchanged for audit/provenance"
            ),
        },
    }

    validate_no_test_fields_in_selection_artifacts(
        {
            "decision": decision_rows,
            "eligibility": eligibility_rows,
            "pareto": pareto_rows,
            "trace": trace_rows,
        },
        protocol,
    )

    atomic_json(OUTPUT_PROTOCOL, protocol)
    atomic_csv(
        OUTPUT_VALIDATION_SOURCE,
        validation_rows,
        [
            "architecture",
            "variant",
            "seed",
            "validation_fingerprint_macro_f1",
            "validation_metric_source",
            "source_file",
            "source_file_sha256",
        ],
    )
    atomic_csv(
        OUTPUT_DECISION,
        decision_rows,
        list(decision_rows[0].keys()),
    )
    atomic_csv(
        OUTPUT_ELIGIBILITY,
        eligibility_rows,
        list(eligibility_rows[0].keys()),
    )
    atomic_csv(
        OUTPUT_PARETO,
        pareto_rows,
        list(pareto_rows[0].keys()),
    )
    atomic_csv(
        OUTPUT_TRACE,
        trace_rows,
        list(trace_rows[0].keys()),
    )

    trace_lookup = {
        row["profile"]: row
        for row in trace_rows
    }
    balanced = trace_lookup[PRIMARY_PROFILE]

    summary = {
        "status": "completed",
        "phase": 8,
        "artifact_name": (
            "validation_only_corrected_multi_objective_selection"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": utc_now(),
        "corrective_reanalysis": True,
        "original_phase8_artifacts_modified": False,
        "test_arrays_loaded": False,
        "test_metrics_used_for_selection": False,
        "phase7_p_values_loaded": False,
        "robustness_used_for_selection": False,
        "validation_source_row_count": len(validation_rows),
        "decision_group_count": len(decision_rows),
        "compressed_candidate_count": 18,
        "profile_results": {
            profile: {
                "eligible_candidate_count": int(
                    row["eligible_candidate_count"]
                ),
                "Pareto_candidate_count": int(
                    row["Pareto_candidate_count"]
                ),
                "selected_candidate_id": row[
                    "selected_candidate_id"
                ],
                "selected_architecture": row[
                    "selected_architecture"
                ],
                "selected_variant": row[
                    "selected_variant"
                ],
            }
            for profile, row in trace_lookup.items()
        },
        "primary_profile": PRIMARY_PROFILE,
        "primary_selection": {
            "candidate_id": balanced["selected_candidate_id"],
            "architecture": balanced["selected_architecture"],
            "variant": balanced["selected_variant"],
            "mean_validation_fingerprint_macro_f1": (
                balanced[
                    "selected_mean_validation_macro_f1"
                ]
            ),
            "mean_serialized_state_dict_bytes": (
                balanced["selected_mean_state_bytes"]
            ),
            "mean_batch_1_median_ms": (
                balanced[
                    "selected_mean_batch_1_median_ms"
                ]
            ),
            "mean_batch_32_samples_per_second": (
                balanced[
                    "selected_mean_batch_32_samples_per_second"
                ]
            ),
        },
        "interpretation": {
            "selection_is_preregistered_original_phase8": False,
            "selection_is_corrective_post_hoc_reanalysis": True,
            "manuscript_must_describe_correction_transparently": True,
            "do_not_assume_original_P50_QAT_selection_persists": True,
        },
        "artifacts": {
            "protocol": file_record(OUTPUT_PROTOCOL),
            "validation_source_matrix": file_record(
                OUTPUT_VALIDATION_SOURCE
            ),
            "decision_matrix": file_record(OUTPUT_DECISION),
            "eligibility": file_record(OUTPUT_ELIGIBILITY),
            "Pareto": file_record(OUTPUT_PARETO),
            "selection_trace": file_record(OUTPUT_TRACE),
        },
        "all_checks_passed": True,
    }
    atomic_json(OUTPUT_SUMMARY, summary)

    lock = {
        "status": "locked",
        "phase": 8,
        "artifact_name": (
            "validation_only_corrected_multi_objective_selection"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "locked_at_utc": utc_now(),
        "primary_profile": PRIMARY_PROFILE,
        "selected_candidate_id": balanced[
            "selected_candidate_id"
        ],
        "test_metrics_used_for_selection": False,
        "phase7_p_values_used_for_selection": False,
        "robustness_used_for_selection": False,
        "original_phase8_artifacts_modified": False,
        "protocol": str(OUTPUT_PROTOCOL),
        "protocol_sha256": sha256_file(OUTPUT_PROTOCOL),
        "summary": str(OUTPUT_SUMMARY),
        "summary_sha256": sha256_file(OUTPUT_SUMMARY),
        "all_checks_passed": True,
    }
    atomic_json(OUTPUT_LOCK, lock)

    source_paths = (
        TINYML_B0,
        COMPACT_B0,
        FP32_FT,
        PRUNING_NOFT,
        PRUNING_FP32_FT,
        DQ_RUNS,
        QAT_RUNS,
        PTQ_SELECTION_LOCK,
        PTQ_VERIFIED_CANDIDATES,
        PHASE6_GROUP,
        PHASE6_LOCK,
    )

    manifest = {
        "status": "locked",
        "phase": 8,
        "artifact_name": (
            "validation_only_corrected_multi_objective_selection"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "locked_at_utc": utc_now(),
        "source_artifacts": [
            file_record(path)
            for path in source_paths
        ],
        "generated_artifacts": [
            file_record(path)
            for path in (
                OUTPUT_PROTOCOL,
                OUTPUT_VALIDATION_SOURCE,
                OUTPUT_DECISION,
                OUTPUT_ELIGIBILITY,
                OUTPUT_PARETO,
                OUTPUT_TRACE,
                OUTPUT_SUMMARY,
                OUTPUT_LOCK,
            )
        ],
        "original_phase8_artifacts_modified": False,
        "all_checks_passed": True,
    }
    atomic_json(OUTPUT_MANIFEST, manifest)

    print("Validation source rows          :", len(validation_rows))
    print("Decision groups                 :", len(decision_rows))
    print("Compressed candidates           : 18")
    print()

    for profile in ("strict", "balanced", "relaxed"):
        row = trace_lookup[profile]
        print(
            f"{profile:<8} | "
            f"eligible={int(row['eligible_candidate_count']):>2} | "
            f"Pareto={int(row['Pareto_candidate_count']):>2} | "
            f"selected={row['selected_candidate_id']}"
        )

    print()
    print("PRIMARY CORRECTED SELECTION")
    print("-" * 96)
    print(
        "Candidate                       :",
        balanced["selected_candidate_id"],
    )
    print(
        "Mean validation Macro-F1        :",
        balanced["selected_mean_validation_macro_f1"],
    )
    print(
        "Mean serialized state bytes     :",
        balanced["selected_mean_state_bytes"],
    )
    print(
        "Mean batch-1 median ms          :",
        balanced["selected_mean_batch_1_median_ms"],
    )
    print(
        "Mean batch-32 samples/s         :",
        balanced[
            "selected_mean_batch_32_samples_per_second"
        ],
    )
    print()
    print("Test metrics used in decision   : False")
    print("Original Phase 8 modified       : False")
    print("All checks passed               : True")
    print(
        "Summary                         :",
        OUTPUT_SUMMARY,
    )
    print(
        "Lock                            :",
        OUTPUT_LOCK,
    )
    print(
        "PHASE 8 VALIDATION-ONLY CORRECTIVE SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
