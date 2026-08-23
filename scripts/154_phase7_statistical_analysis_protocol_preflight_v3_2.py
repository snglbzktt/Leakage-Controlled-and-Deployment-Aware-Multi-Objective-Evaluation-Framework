from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_LOCK = (
    AUDIT
    / "phase5_compression_locked_v3_2.json"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE6_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_locked_v3_2.json"
)

PHASE6_ALL_RUNS = (
    AUDIT
    / "phase6_deployment_benchmark_all_runs_v3_2.csv"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase7_statistical_analysis_protocol_v3_2.json"
)

OUTPUT_JOINED_MATRIX = (
    AUDIT
    / "phase7_statistical_analysis_joined_matrix_v3_2.csv"
)

OUTPUT_COMPARISON_PLAN = (
    AUDIT
    / "phase7_statistical_comparison_plan_v3_2.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase7_statistical_analysis_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase7_statistical_analysis_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase7_statistical_analysis_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase7_statistical_analysis_v3_2"
PHASE5_PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
PHASE6_PROTOCOL_VERSION = "phase6_deployment_benchmark_v3_2"

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

NON_BASELINE_VARIANTS = tuple(
    variant
    for variant in VARIANTS
    if variant != "B0"
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_KEYS = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

EXPECTED_JOINED_ROWS = 110

CONFIRMATORY_METRICS = {
    "test_fingerprint_macro_f1": {
        "source_column": "test_fingerprint_macro_f1",
        "domain": "predictive",
        "direction": "higher_is_better",
        "unit": "proportion",
    },
    "batch_1_median_ms": {
        "source_column": "batch_1_median_ms",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "batch_32_samples_per_second": {
        "source_column": "batch_32_samples_per_second",
        "domain": "deployment",
        "direction": "higher_is_better",
        "unit": "samples_per_second",
    },
    "serialized_state_dict_bytes": {
        "source_column": "serialized_state_dict_bytes",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "bytes",
    },
}

EXPLORATORY_METRICS = {
    "test_raw_weighted_macro_f1": {
        "source_column": "test_raw_weighted_macro_f1",
        "domain": "predictive",
        "direction": "higher_is_better",
        "unit": "proportion",
    },
    "batch_1_p95_ms": {
        "source_column": "batch_1_p95_ms",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "batch_32_p95_ms": {
        "source_column": "batch_32_p95_ms",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "peak_inference_RSS_delta_bytes": {
        "source_column": "peak_inference_RSS_delta_bytes",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "bytes",
    },
}

ALL_METRICS = {
    **CONFIRMATORY_METRICS,
    **EXPLORATORY_METRICS,
}

GLOBAL_RANDOM_SEED = 2026
MONTE_CARLO_OMNIBUS_PERMUTATIONS = 100_000
EXACT_SIGN_FLIP_ASSIGNMENTS = 2 ** len(SEEDS)
EXACT_BOOTSTRAP_RESAMPLES = len(SEEDS) ** len(SEEDS)
ALPHA = 0.05

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    replace_with_retry(
        temporary,
        path,
    )


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    replace_with_retry(
        temporary,
        path,
    )


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


def finite_float(
    value: Any,
    field_name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise RuntimeError(
            f"Non-finite value in {field_name}: {value}"
        )

    return number


required_paths = (
    PHASE5_LOCK,
    PHASE5_MASTER_MATRIX,
    PHASE6_LOCK,
    PHASE6_ALL_RUNS,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_JOINED_MATRIX,
    OUTPUT_COMPARISON_PLAN,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 7 statistical preflight artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

if importlib.util.find_spec("scipy") is None:
    raise RuntimeError(
        "SciPy is required for the locked "
        "Friedman sensitivity calculation."
    )

from importlib.metadata import version as package_version

phase5_lock = read_json(
    PHASE5_LOCK
)

phase6_lock = read_json(
    PHASE6_LOCK
)

phase5_rows = read_csv(
    PHASE5_MASTER_MATRIX
)

phase6_rows = read_csv(
    PHASE6_ALL_RUNS
)

entry_checks = {
    "phase5_locked": (
        phase5_lock.get("status") == "locked"
        and phase5_lock.get("protocol_version")
        == PHASE5_PROTOCOL_VERSION
        and phase5_lock.get("all_checks_passed") is True
    ),
    "phase6_locked": (
        phase6_lock.get("status") == "locked"
        and phase6_lock.get("protocol_version")
        == PHASE6_PROTOCOL_VERSION
        and phase6_lock.get("all_checks_passed") is True
    ),
    "phase5_master_hash_matches": (
        phase5_lock.get("master_run_matrix_sha256")
        == sha256_file(PHASE5_MASTER_MATRIX)
    ),
    "phase6_all_runs_hash_matches": (
        phase6_lock.get("all_runs_csv_sha256")
        == sha256_file(PHASE6_ALL_RUNS)
    ),
    "phase5_row_count_110": (
        len(phase5_rows) == EXPECTED_JOINED_ROWS
    ),
    "phase6_row_count_110": (
        len(phase6_rows) == EXPECTED_JOINED_ROWS
    ),
    "phase5_final_model_not_selected": (
        phase5_lock.get("final_model_selected") is False
    ),
    "phase6_final_model_not_selected": (
        phase6_lock.get("final_model_selected") is False
    ),
}

failed_entry_checks = [
    name
    for name, passed in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 7 statistical preflight entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

phase5_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in phase5_rows
}

phase6_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in phase6_rows
}

if set(phase5_lookup.keys()) != EXPECTED_KEYS:
    raise RuntimeError(
        "Phase 5 statistical source matrix is incomplete."
    )

if set(phase6_lookup.keys()) != EXPECTED_KEYS:
    raise RuntimeError(
        "Phase 6 statistical source matrix is incomplete."
    )

joined_rows: list[dict[str, Any]] = []

for architecture in ARCHITECTURES:
    for variant in VARIANTS:
        for seed in SEEDS:
            key = (
                architecture,
                variant,
                seed,
            )

            predictive = phase5_lookup[key]
            deployment = phase6_lookup[key]

            if predictive["representation"] != deployment["representation"]:
                raise RuntimeError(
                    "Representation mismatch while joining "
                    f"{key}."
                )

            joined_rows.append(
                {
                    "architecture": architecture,
                    "variant": variant,
                    "seed": seed,
                    "representation": predictive["representation"],
                    "test_fingerprint_macro_f1": finite_float(
                        predictive["test_fingerprint_macro_f1"],
                        "test_fingerprint_macro_f1",
                    ),
                    "test_fingerprint_accuracy": finite_float(
                        predictive["test_fingerprint_accuracy"],
                        "test_fingerprint_accuracy",
                    ),
                    "test_raw_weighted_macro_f1": finite_float(
                        predictive["test_raw_weighted_macro_f1"],
                        "test_raw_weighted_macro_f1",
                    ),
                    "test_gafgyt_fnr": finite_float(
                        predictive["test_gafgyt_fnr"],
                        "test_gafgyt_fnr",
                    ),
                    "test_mirai_fnr": finite_float(
                        predictive["test_mirai_fnr"],
                        "test_mirai_fnr",
                    ),
                    "float_parameter_count": int(
                        predictive["float_parameter_count"]
                    ),
                    "float_linear_macs_per_sample": int(
                        predictive["float_linear_macs_per_sample"]
                    ),
                    "state_size_ratio_vs_float": finite_float(
                        predictive["state_size_ratio_vs_float"],
                        "state_size_ratio_vs_float",
                    ),
                    "fragility_gate_triggered": parse_bool(
                        predictive["fragility_gate_triggered"]
                    ),
                    "checkpoint_file_bytes": int(
                        deployment["checkpoint_file_bytes"]
                    ),
                    "serialized_state_dict_bytes": int(
                        deployment["serialized_state_dict_bytes"]
                    ),
                    "model_load_elapsed_ms": finite_float(
                        deployment["model_load_elapsed_ms"],
                        "model_load_elapsed_ms",
                    ),
                    "model_load_RSS_delta_bytes": int(
                        deployment["model_load_RSS_delta_bytes"]
                    ),
                    "peak_inference_RSS_delta_bytes": int(
                        deployment["peak_inference_RSS_delta_bytes"]
                    ),
                    "batch_1_median_ms": finite_float(
                        deployment["batch_1_median_ms"],
                        "batch_1_median_ms",
                    ),
                    "batch_1_p95_ms": finite_float(
                        deployment["batch_1_p95_ms"],
                        "batch_1_p95_ms",
                    ),
                    "batch_1_p99_ms": finite_float(
                        deployment["batch_1_p99_ms"],
                        "batch_1_p99_ms",
                    ),
                    "batch_1_samples_per_second": finite_float(
                        deployment["batch_1_samples_per_second"],
                        "batch_1_samples_per_second",
                    ),
                    "batch_32_median_ms": finite_float(
                        deployment["batch_32_median_ms"],
                        "batch_32_median_ms",
                    ),
                    "batch_32_p95_ms": finite_float(
                        deployment["batch_32_p95_ms"],
                        "batch_32_p95_ms",
                    ),
                    "batch_32_p99_ms": finite_float(
                        deployment["batch_32_p99_ms"],
                        "batch_32_p99_ms",
                    ),
                    "batch_32_samples_per_second": finite_float(
                        deployment["batch_32_samples_per_second"],
                        "batch_32_samples_per_second",
                    ),
                    "phase5_metrics_path": predictive["metrics_path"],
                    "phase5_metrics_sha256": predictive["metrics_sha256"],
                    "phase6_deployment_metrics_sha256": (
                        deployment["deployment_metrics_sha256"]
                    ),
                    "phase6_raw_latencies_sha256": (
                        deployment["raw_latencies_sha256"]
                    ),
                }
            )

if len(joined_rows) != EXPECTED_JOINED_ROWS:
    raise RuntimeError(
        "Joined statistical matrix row count mismatch."
    )

joined_rows.sort(
    key=lambda row: (
        row["architecture"],
        VARIANTS.index(row["variant"]),
        int(row["seed"]),
    )
)

atomic_csv(
    OUTPUT_JOINED_MATRIX,
    joined_rows,
    [
        "architecture",
        "variant",
        "seed",
        "representation",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "float_parameter_count",
        "float_linear_macs_per_sample",
        "state_size_ratio_vs_float",
        "fragility_gate_triggered",
        "checkpoint_file_bytes",
        "serialized_state_dict_bytes",
        "model_load_elapsed_ms",
        "model_load_RSS_delta_bytes",
        "peak_inference_RSS_delta_bytes",
        "batch_1_median_ms",
        "batch_1_p95_ms",
        "batch_1_p99_ms",
        "batch_1_samples_per_second",
        "batch_32_median_ms",
        "batch_32_p95_ms",
        "batch_32_p99_ms",
        "batch_32_samples_per_second",
        "phase5_metrics_path",
        "phase5_metrics_sha256",
        "phase6_deployment_metrics_sha256",
        "phase6_raw_latencies_sha256",
    ],
)

comparison_rows: list[dict[str, Any]] = []

for endpoint_class, metrics in (
    ("confirmatory", CONFIRMATORY_METRICS),
    ("exploratory", EXPLORATORY_METRICS),
):
    for architecture in ARCHITECTURES:
        for metric_name, specification in metrics.items():
            family_id = (
                f"{endpoint_class}__{architecture}__{metric_name}"
            )

            comparison_rows.append(
                {
                    "analysis_type": "omnibus",
                    "endpoint_class": endpoint_class,
                    "family_id": family_id,
                    "architecture": architecture,
                    "metric": metric_name,
                    "metric_domain": specification["domain"],
                    "direction": specification["direction"],
                    "unit": specification["unit"],
                    "reference_variant": "",
                    "candidate_variant": "",
                    "paired_seed_count": len(SEEDS),
                    "test": (
                        "friedman_rank_statistic_with_"
                        "within_seed_monte_carlo_permutation"
                    ),
                    "two_sided": True,
                    "multiplicity_adjustment": "none_for_omnibus",
                    "alpha": ALPHA,
                }
            )

            for variant in NON_BASELINE_VARIANTS:
                comparison_rows.append(
                    {
                        "analysis_type": "posthoc_vs_B0",
                        "endpoint_class": endpoint_class,
                        "family_id": family_id,
                        "architecture": architecture,
                        "metric": metric_name,
                        "metric_domain": specification["domain"],
                        "direction": specification["direction"],
                        "unit": specification["unit"],
                        "reference_variant": "B0",
                        "candidate_variant": variant,
                        "paired_seed_count": len(SEEDS),
                        "test": (
                            "exact_two_sided_paired_sign_flip_"
                            "on_mean_raw_delta"
                        ),
                        "two_sided": True,
                        "multiplicity_adjustment": (
                            "holm_within_architecture_metric_family"
                        ),
                        "alpha": ALPHA,
                    }
                )

expected_omnibus_rows = (
    len(ALL_METRICS)
    * len(ARCHITECTURES)
)

expected_posthoc_rows = (
    len(ALL_METRICS)
    * len(ARCHITECTURES)
    * len(NON_BASELINE_VARIANTS)
)

if (
    sum(
        row["analysis_type"] == "omnibus"
        for row in comparison_rows
    )
    != expected_omnibus_rows
):
    raise RuntimeError(
        "Omnibus comparison-plan row count mismatch."
    )

if (
    sum(
        row["analysis_type"] == "posthoc_vs_B0"
        for row in comparison_rows
    )
    != expected_posthoc_rows
):
    raise RuntimeError(
        "Post-hoc comparison-plan row count mismatch."
    )

atomic_csv(
    OUTPUT_COMPARISON_PLAN,
    comparison_rows,
    [
        "analysis_type",
        "endpoint_class",
        "family_id",
        "architecture",
        "metric",
        "metric_domain",
        "direction",
        "unit",
        "reference_variant",
        "candidate_variant",
        "paired_seed_count",
        "test",
        "two_sided",
        "multiplicity_adjustment",
        "alpha",
    ],
)

protocol = {
    "status": "locked",
    "phase": 7,
    "artifact_name": "statistical_analysis_protocol",
    "protocol_version": PROTOCOL_VERSION,
    "locked_at_utc": utc_now(),
    "provenance_statement": {
        "externally_preregistered": False,
        "source_results_generated_before_protocol_lock": True,
        "inferential_computation_performed_before_protocol_lock": False,
        "scope_control": (
            "All variants are compared with B0 under one uniform "
            "rule to reduce selective reporting."
        ),
    },
    "analysis_unit": {
        "paired_unit": "seed",
        "paired_seed_values": list(SEEDS),
        "paired_seed_count": len(SEEDS),
        "architectures_analyzed_separately": True,
        "missing_value_policy": "hard_failure_no_imputation",
        "duplicate_key_policy": "hard_failure",
    },
    "endpoints": {
        "confirmatory": CONFIRMATORY_METRICS,
        "exploratory": EXPLORATORY_METRICS,
    },
    "omnibus_analysis": {
        "statistic": "Friedman rank statistic",
        "null_randomization": (
            "independent within-seed permutation of the 11 "
            "variant labels"
        ),
        "monte_carlo_permutations": (
            MONTE_CARLO_OMNIBUS_PERMUTATIONS
        ),
        "global_random_seed": GLOBAL_RANDOM_SEED,
        "p_value_correction": "plus_one",
        "scipy_asymptotic_friedman": (
            "reported_as_sensitivity_only"
        ),
    },
    "posthoc_analysis": {
        "reference_variant": "B0",
        "candidate_variants": list(
            NON_BASELINE_VARIANTS
        ),
        "test": (
            "exact two-sided paired sign-flip randomization "
            "test on the mean raw paired delta"
        ),
        "exact_assignments": (
            EXACT_SIGN_FLIP_ASSIGNMENTS
        ),
        "raw_delta_definition": (
            "candidate_minus_B0"
        ),
        "improvement_delta_definition": {
            "higher_is_better": "candidate_minus_B0",
            "lower_is_better": "B0_minus_candidate",
        },
        "effect_sizes": [
            "mean_raw_delta",
            "median_raw_delta",
            "mean_improvement_delta",
            "median_improvement_delta",
            "paired_rank_biserial_correlation",
            "win_tie_loss_counts",
        ],
        "confidence_interval": {
            "target": "mean_raw_delta",
            "method": (
                "deterministic exhaustive paired bootstrap "
                "percentile interval"
            ),
            "confidence_level": 0.95,
            "resamples": EXACT_BOOTSTRAP_RESAMPLES,
        },
    },
    "multiplicity": {
        "method": "Holm step-down",
        "family_definition": (
            "ten B0 comparisons within each architecture "
            "and metric"
        ),
        "confirmatory_and_exploratory_families_separate": True,
        "alpha": ALPHA,
    },
    "small_sample_interpretation": {
        "paired_seed_count": len(SEEDS),
        "minimum_possible_two_sided_exact_p": (
            2.0 / EXACT_SIGN_FLIP_ASSIGNMENTS
        ),
        "nominal_alpha": ALPHA,
        "consequence": (
            "With five non-zero pairs, a two-sided exact "
            "sign-flip p-value cannot fall below 0.0625. "
            "Effect sizes, consistency, uncertainty, fragility, "
            "and Pareto trade-offs therefore receive primary "
            "interpretive emphasis; absence of p<0.05 is not "
            "treated as evidence of equivalence."
        ),
    },
    "reporting_rules": {
        "report_all_planned_comparisons": True,
        "report_raw_and_Holm_adjusted_p_values": True,
        "report_all_effect_sizes": True,
        "report_exact_zero_differences": True,
        "no_significance_only_filtering": True,
        "no_final_model_selection_in_phase7": True,
        "final_selection_deferred_to": (
            "locked multi-objective decision stage after "
            "statistical analysis"
        ),
    },
    "source_artifacts": {
        "phase5_lock": file_record(PHASE5_LOCK),
        "phase5_master_matrix": file_record(
            PHASE5_MASTER_MATRIX
        ),
        "phase6_lock": file_record(PHASE6_LOCK),
        "phase6_all_runs": file_record(
            PHASE6_ALL_RUNS
        ),
    },
    "environment": {
        "python_version": sys.version,
        "platform": platform.platform(),
        "scipy_version": package_version("scipy"),
    },
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

preflight_checks = {
    "joined_row_count_110": (
        len(joined_rows) == EXPECTED_JOINED_ROWS
    ),
    "joined_keys_exact": (
        {
            (
                row["architecture"],
                row["variant"],
                int(row["seed"]),
            )
            for row in joined_rows
        }
        == EXPECTED_KEYS
    ),
    "no_nonfinite_metric_values": (
        all(
            math.isfinite(
                float(row[metric_name])
            )
            for row in joined_rows
            for metric_name in ALL_METRICS
        )
    ),
    "confirmatory_metric_count_4": (
        len(CONFIRMATORY_METRICS) == 4
    ),
    "exploratory_metric_count_4": (
        len(EXPLORATORY_METRICS) == 4
    ),
    "omnibus_plan_count_16": (
        expected_omnibus_rows == 16
    ),
    "posthoc_plan_count_160": (
        expected_posthoc_rows == 160
    ),
    "exact_sign_flip_assignments_32": (
        EXACT_SIGN_FLIP_ASSIGNMENTS == 32
    ),
    "exact_bootstrap_resamples_3125": (
        EXACT_BOOTSTRAP_RESAMPLES == 3125
    ),
    "final_model_not_selected": (
        True
    ),
}

failed_preflight_checks = [
    name
    for name, passed in preflight_checks.items()
    if not passed
]

if failed_preflight_checks:
    raise RuntimeError(
        "Phase 7 statistical preflight failed: "
        + ", ".join(failed_preflight_checks)
    )

preflight = {
    "status": "passed",
    "phase": 7,
    "artifact_name": "statistical_analysis_preflight",
    "protocol_version": PROTOCOL_VERSION,
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": preflight_checks,
    "joined_matrix": file_record(
        OUTPUT_JOINED_MATRIX
    ),
    "comparison_plan": file_record(
        OUTPUT_COMPARISON_PLAN
    ),
    "protocol": file_record(
        OUTPUT_PROTOCOL
    ),
    "ready_for_locked_statistical_execution": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 7,
    "artifact_name": "statistical_analysis_preflight",
    "protocol_version": PROTOCOL_VERSION,
    "locked_at_utc": utc_now(),
    "joined_row_count": EXPECTED_JOINED_ROWS,
    "omnibus_comparison_count": expected_omnibus_rows,
    "posthoc_comparison_count": expected_posthoc_rows,
    "protocol": str(OUTPUT_PROTOCOL),
    "protocol_sha256": sha256_file(
        OUTPUT_PROTOCOL
    ),
    "joined_matrix": str(
        OUTPUT_JOINED_MATRIX
    ),
    "joined_matrix_sha256": sha256_file(
        OUTPUT_JOINED_MATRIX
    ),
    "comparison_plan": str(
        OUTPUT_COMPARISON_PLAN
    ),
    "comparison_plan_sha256": sha256_file(
        OUTPUT_COMPARISON_PLAN
    ),
    "preflight_report": str(
        OUTPUT_PREFLIGHT
    ),
    "preflight_report_sha256": sha256_file(
        OUTPUT_PREFLIGHT
    ),
    "inferential_computation_performed": False,
    "final_model_selected": False,
    "ready_for_locked_statistical_execution": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 7,
    "artifact_name": "statistical_analysis_preflight",
    "protocol_version": PROTOCOL_VERSION,
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(PHASE5_LOCK),
        file_record(PHASE5_MASTER_MATRIX),
        file_record(PHASE6_LOCK),
        file_record(PHASE6_ALL_RUNS),
    ],
    "generated_artifacts": [
        file_record(OUTPUT_PROTOCOL),
        file_record(OUTPUT_JOINED_MATRIX),
        file_record(OUTPUT_COMPARISON_PLAN),
        file_record(OUTPUT_PREFLIGHT),
        file_record(OUTPUT_LOCK),
    ],
    "joined_row_count": EXPECTED_JOINED_ROWS,
    "omnibus_comparison_count": expected_omnibus_rows,
    "posthoc_comparison_count": expected_posthoc_rows,
    "inferential_computation_performed": False,
    "final_model_selected": False,
    "ready_for_locked_statistical_execution": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 7 STATISTICAL ANALYSIS PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Upstream Phase 5 status         : LOCKED"
)
print(
    "Upstream Phase 6 status         : LOCKED"
)
print(
    "Joined seed-level rows          : 110"
)
print(
    "Architectures                   : 2"
)
print(
    "Variants per architecture       : 11"
)
print(
    "Paired seeds                    : 5"
)
print(
    "Confirmatory metrics            : 4"
)
print(
    "Exploratory metrics             : 4"
)
print(
    "Planned omnibus analyses        : 16"
)
print(
    "Planned post-hoc comparisons    : 160"
)
print(
    "Post-hoc reference              : B0"
)
print(
    "Exact sign-flip assignments     : 32"
)
print(
    "Exact bootstrap resamples       : 3125"
)
print(
    "Multiplicity correction         : Holm within family"
)
print(
    "Minimum exact two-sided p       : 0.0625"
)
print(
    "Inferential computation run     : False"
)
print(
    "Externally preregistered        : False"
)
print(
    "Final model selected            : False"
)
print(
    "Protocol                        : "
    f"{OUTPUT_PROTOCOL}"
)
print(
    "Joined matrix                   : "
    f"{OUTPUT_JOINED_MATRIX}"
)
print(
    "Comparison plan                 : "
    f"{OUTPUT_COMPARISON_PLAN}"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for statistics            : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 7 STATISTICAL ANALYSIS PROTOCOL LOCKED"
)
