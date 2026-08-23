from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "early_lodo_global_gate_v2_1"
SEED = 2026
SEVERE_MACRO_F1_THRESHOLD = 0.85
SEVERE_ATTACK_FNR_THRESHOLD = 0.40
MEANINGFUL_DROP_THRESHOLD = 0.10

EXPECTED_DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

EXPECTED_MODELS = [
    "tinyml_mlp_b0",
    "compact_dnn_b0",
    "hist_gradient_boosting_b0",
]

TWO_CLASS_TEST_DEVICES = {
    "Ennio_Doorbell",
    "Samsung_SNH_1011_N_Webcam",
}

ROOT = Path(__file__).resolve().parents[1]
EARLY_LODO_DIR = ROOT / "results" / "v2" / "early_lodo"
DEVICES_DIR = EARLY_LODO_DIR / "devices"
DEVICE_LEDGER = EARLY_LODO_DIR / "early_lodo_device_ledger_v2.csv"
GLOBAL_DIR = EARLY_LODO_DIR / "global"
DOC_DIR = ROOT / "docs" / "v2" / "early_lodo"

GLOBAL_SUMMARY = GLOBAL_DIR / "early_lodo_global_summary_v2.json"
MODEL_SUMMARY_CSV = GLOBAL_DIR / "early_lodo_model_summary_v2.csv"
MATRIX_CSV = GLOBAL_DIR / "early_lodo_device_model_matrix_v2.csv"
REPORT_TXT = GLOBAL_DIR / "early_lodo_global_report_v2.txt"
COMPLETE_NOTE = DOC_DIR / "EARLY_LODO_GLOBAL_COMPLETE.md"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def fmt(value: Any, digits: int = 9) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def project_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    normalized = raw.replace("\\", "/")
    return ROOT / Path(normalized)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    summary_files = sorted(DEVICES_DIR.glob("*/device_summary.json"))
    summaries = [read_json(path) for path in summary_files]
    summaries_by_device = {
        str(summary.get("held_out_device")): summary for summary in summaries
    }

    device_ledger_rows: list[dict[str, str]] = []
    if DEVICE_LEDGER.exists():
        with DEVICE_LEDGER.open("r", encoding="utf-8-sig", newline="") as handle:
            device_ledger_rows = list(csv.DictReader(handle))

    model_rows: list[dict[str, Any]] = []
    severe_cells: list[dict[str, Any]] = []
    meaningful_drop_cells: list[dict[str, Any]] = []
    local_flag_matches: list[bool] = []
    lock_files_exist: list[bool] = []
    result_files_exist: list[bool] = []
    device_ranking_valid: list[bool] = []
    class_coverage_valid: list[bool] = []

    for device in EXPECTED_DEVICES:
        summary = summaries_by_device.get(device)
        if summary is None:
            continue

        test_class_counts = [
            int(value) for value in summary.get("device_data", {}).get(
                "test_class_counts", []
            )
        ]
        present_class_count = sum(value > 0 for value in test_class_counts)
        expected_present_count = 2 if device in TWO_CLASS_TEST_DEVICES else 3
        class_coverage_valid.append(present_class_count == expected_present_count)

        results = summary.get("model_results", [])
        if isinstance(results, list) and results:
            expected_best = max(
                results,
                key=lambda item: float(item["test_fingerprint_macro_f1"]),
            )["model_id"]
            actual_best = summary.get("ranking", {}).get(
                "best_test_macro_f1_model"
            )
            device_ranking_valid.append(actual_best == expected_best)
        else:
            device_ranking_valid.append(False)

        for item in results:
            macro_f1 = float(item["test_fingerprint_macro_f1"])
            accuracy = float(item["test_fingerprint_accuracy"])
            record_macro_f1 = float(item["test_record_weighted_macro_f1"])
            validation_macro_f1 = float(item["validation_macro_f1"])
            validation_drop = float(item["validation_test_macro_f1_drop"])
            runtime = float(item["total_seconds"])

            benign_fnr = item.get("benign_fnr")
            gafgyt_fnr = item.get("gafgyt_fnr")
            mirai_fnr = item.get("mirai_fnr")

            eligible_attack_fnrs = [
                float(value)
                for value in (gafgyt_fnr, mirai_fnr)
                if value is not None
            ]
            maximum_attack_fnr = (
                max(eligible_attack_fnrs) if eligible_attack_fnrs else None
            )

            recomputed_severe = (
                macro_f1 < SEVERE_MACRO_F1_THRESHOLD
                or (
                    maximum_attack_fnr is not None
                    and maximum_attack_fnr > SEVERE_ATTACK_FNR_THRESHOLD
                )
            )
            stored_severe = bool(item.get("local_severe_signal"))
            local_flag_matches.append(recomputed_severe == stored_severe)

            row = {
                "held_out_device": device,
                "model_id": str(item["model_id"]),
                "run_id": str(item["run_id"]),
                "seed": int(item["seed"]),
                "test_present_class_count": present_class_count,
                "test_class_counts": "|".join(str(value) for value in test_class_counts),
                "validation_macro_f1": validation_macro_f1,
                "test_fingerprint_macro_f1": macro_f1,
                "test_fingerprint_accuracy": accuracy,
                "test_record_weighted_macro_f1": record_macro_f1,
                "validation_test_macro_f1_drop": validation_drop,
                "benign_fnr": None if benign_fnr is None else float(benign_fnr),
                "gafgyt_fnr": None if gafgyt_fnr is None else float(gafgyt_fnr),
                "mirai_fnr": None if mirai_fnr is None else float(mirai_fnr),
                "maximum_eligible_attack_fnr": maximum_attack_fnr,
                "local_severe_signal": recomputed_severe,
                "meaningful_validation_test_drop": (
                    validation_drop >= MEANINGFUL_DROP_THRESHOLD
                ),
                "parameter_count": item.get("parameter_count"),
                "completed_epoch": item.get("completed_epoch"),
                "best_epoch": item.get("best_epoch"),
                "stopped_early": item.get("stopped_early"),
                "completed_iterations": item.get("completed_iterations"),
                "total_seconds": runtime,
                "lock_summary_file": str(item["lock_summary_file"]),
                "run_result_file": str(item["run_result_file"]),
            }
            model_rows.append(row)

            lock_files_exist.append(
                project_path(str(item["lock_summary_file"])).exists()
            )
            result_files_exist.append(
                project_path(str(item["run_result_file"])).exists()
            )

            if recomputed_severe:
                severe_cells.append(row)
            if validation_drop >= MEANINGFUL_DROP_THRESHOLD:
                meaningful_drop_cells.append(row)

    run_ids = [row["run_id"] for row in model_rows]
    device_model_keys = [
        (row["held_out_device"], row["model_id"]) for row in model_rows
    ]

    ledger_devices = {row.get("held_out_device", "") for row in device_ledger_rows}
    ledger_status_valid = all(
        row.get("status") == "complete"
        and row.get("completed_model_count") == "3"
        and row.get("seed") == str(SEED)
        for row in device_ledger_rows
        if row.get("held_out_device") in EXPECTED_DEVICES
    )

    validation_checks = {
        "exactly_nine_device_summaries": len(summaries) == 9,
        "expected_devices_present": set(summaries_by_device) == set(EXPECTED_DEVICES),
        "all_summaries_complete": all(
            summary.get("status") == "complete" for summary in summaries
        ),
        "all_summary_checks_passed": all(
            summary.get("all_checks_passed") is True for summary in summaries
        ),
        "all_seed_2026": all(
            int(summary.get("seed", -1)) == SEED for summary in summaries
        ),
        "all_three_models": all(
            int(summary.get("completed_model_count", -1)) == 3
            for summary in summaries
        ),
        "expected_model_set_per_device": all(
            set(summary.get("expected_models", [])) == set(EXPECTED_MODELS)
            and {
                item.get("model_id")
                for item in summary.get("model_results", [])
            }
            == set(EXPECTED_MODELS)
            for summary in summaries
        ),
        "exactly_27_model_rows": len(model_rows) == 27,
        "unique_27_run_ids": len(run_ids) == 27 and len(set(run_ids)) == 27,
        "unique_device_model_cells": (
            len(device_model_keys) == 27
            and len(set(device_model_keys)) == 27
        ),
        "all_primary_metrics_finite": all(
            all(
                is_finite_number(row[field])
                for field in (
                    "validation_macro_f1",
                    "test_fingerprint_macro_f1",
                    "test_fingerprint_accuracy",
                    "test_record_weighted_macro_f1",
                    "validation_test_macro_f1_drop",
                )
            )
            for row in model_rows
        ),
        "all_runtime_positive": all(
            is_finite_number(row["total_seconds"])
            and float(row["total_seconds"]) > 0.0
            for row in model_rows
        ),
        "all_test_counts_positive": all(
            int(row["test_present_class_count"]) >= 2 for row in model_rows
        ),
        "class_coverage_matches_protocol": all(class_coverage_valid)
        and len(class_coverage_valid) == 9,
        "all_local_severe_flags_match": all(local_flag_matches)
        and len(local_flag_matches) == 27,
        "all_device_rankings_valid": all(device_ranking_valid)
        and len(device_ranking_valid) == 9,
        "all_lock_files_exist": all(lock_files_exist)
        and len(lock_files_exist) == 27,
        "all_run_result_files_exist": all(result_files_exist)
        and len(result_files_exist) == 27,
        "device_ledger_exists": DEVICE_LEDGER.exists(),
        "device_ledger_has_nine_rows": len(device_ledger_rows) == 9,
        "device_ledger_expected_devices": ledger_devices == set(EXPECTED_DEVICES),
        "device_ledger_rows_complete": ledger_status_valid
        and len(device_ledger_rows) == 9,
    }

    failed_checks = [
        name for name, passed in validation_checks.items() if not passed
    ]
    if failed_checks:
        print("=" * 86)
        print("EARLY LODO GLOBAL GATE VALIDATION FAILED")
        print("=" * 86)
        for name in failed_checks:
            print(f"{name}: False")
        return 1

    model_summaries: list[dict[str, Any]] = []
    device_winner_counts = Counter()
    fastest_counts = Counter()

    for summary in summaries:
        device_winner_counts[
            summary["ranking"]["best_test_macro_f1_model"]
        ] += 1
        fastest_counts[summary["ranking"]["fastest_model"]] += 1

    for model_id in EXPECTED_MODELS:
        rows = [row for row in model_rows if row["model_id"] == model_id]
        macro_values = [
            float(row["test_fingerprint_macro_f1"]) for row in rows
        ]
        accuracy_values = [
            float(row["test_fingerprint_accuracy"]) for row in rows
        ]
        record_values = [
            float(row["test_record_weighted_macro_f1"]) for row in rows
        ]
        runtime_values = [float(row["total_seconds"]) for row in rows]
        validation_drop_values = [
            float(row["validation_test_macro_f1_drop"]) for row in rows
        ]
        attack_fnr_values = [
            float(row["maximum_eligible_attack_fnr"])
            for row in rows
            if row["maximum_eligible_attack_fnr"] is not None
        ]

        model_summaries.append(
            {
                "model_id": model_id,
                "device_count": len(rows),
                "mean_test_macro_f1": mean(macro_values),
                "median_test_macro_f1": median(macro_values),
                "sample_std_test_macro_f1": sample_std(macro_values),
                "minimum_test_macro_f1": min(macro_values),
                "maximum_test_macro_f1": max(macro_values),
                "mean_test_accuracy": mean(accuracy_values),
                "mean_record_weighted_macro_f1": mean(record_values),
                "mean_validation_test_macro_f1_drop": mean(
                    validation_drop_values
                ),
                "maximum_validation_test_macro_f1_drop": max(
                    validation_drop_values
                ),
                "maximum_eligible_attack_fnr": max(attack_fnr_values),
                "mean_total_seconds": mean(runtime_values),
                "median_total_seconds": median(runtime_values),
                "device_macro_f1_win_count": device_winner_counts[model_id],
                "device_fastest_count": fastest_counts[model_id],
                "local_severe_cell_count": sum(
                    bool(row["local_severe_signal"]) for row in rows
                ),
                "meaningful_validation_test_drop_count": sum(
                    bool(row["meaningful_validation_test_drop"])
                    for row in rows
                ),
            }
        )

    model_summaries.sort(
        key=lambda row: (
            -float(row["mean_test_macro_f1"]),
            float(row["mean_total_seconds"]),
        )
    )
    for index, row in enumerate(model_summaries, start=1):
        row["global_macro_f1_rank"] = index

    all_macro_values = [
        float(row["test_fingerprint_macro_f1"]) for row in model_rows
    ]
    all_attack_fnr_values = [
        float(row["maximum_eligible_attack_fnr"])
        for row in model_rows
        if row["maximum_eligible_attack_fnr"] is not None
    ]

    gate_passed = len(severe_cells) == 0
    gate_status = (
        "PASS_NO_SEVERE_FRAGILITY"
        if gate_passed
        else "FRAGILITY_DETECTED"
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    global_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "generated_at": generated_at,
        "seed": SEED,
        "scope": {
            "device_count": 9,
            "model_count": 3,
            "scientific_run_count": 27,
            "expected_devices": EXPECTED_DEVICES,
            "expected_models": EXPECTED_MODELS,
            "two_class_test_devices": sorted(TWO_CLASS_TEST_DEVICES),
        },
        "locked_thresholds": {
            "severe_macro_f1_below": SEVERE_MACRO_F1_THRESHOLD,
            "severe_attack_fnr_above": SEVERE_ATTACK_FNR_THRESHOLD,
            "meaningful_validation_test_macro_f1_drop_at_least": (
                MEANINGFUL_DROP_THRESHOLD
            ),
        },
        "global_statistics": {
            "mean_cell_test_macro_f1": mean(all_macro_values),
            "median_cell_test_macro_f1": median(all_macro_values),
            "minimum_cell_test_macro_f1": min(all_macro_values),
            "maximum_cell_test_macro_f1": max(all_macro_values),
            "maximum_eligible_attack_fnr": max(all_attack_fnr_values),
            "local_severe_cell_count": len(severe_cells),
            "meaningful_validation_test_drop_cell_count": len(
                meaningful_drop_cells
            ),
        },
        "model_ranking_basis": (
            "Unweighted mean of per-device test fingerprint Macro-F1 "
            "across all nine held-out devices."
        ),
        "model_ranking": model_summaries,
        "gate": {
            "status": gate_status,
            "passed": gate_passed,
            "rule": (
                "A cell is severe when test Macro-F1 is below 0.85 "
                "or any eligible attack FNR is above 0.40."
            ),
            "severe_cells": severe_cells,
        },
        "interpretation": {
            "general_model_ranking": "permitted_after_9_of_9_devices_complete",
            "test_selection_use": "prohibited",
            "class_coverage_note": (
                "Ennio and Samsung have no Mirai samples in held-out test; "
                "their test Macro-F1 is computed over the classes present."
            ),
            "ranking_caution": (
                "The ranking is an unweighted device-level summary and must "
                "be interpreted with the two-class test coverage note."
            ),
        },
        "validation_checks": validation_checks,
        "all_checks_passed": all(validation_checks.values()),
        "outputs": {
            "global_summary": str(GLOBAL_SUMMARY.relative_to(ROOT)),
            "model_summary_csv": str(MODEL_SUMMARY_CSV.relative_to(ROOT)),
            "device_model_matrix_csv": str(MATRIX_CSV.relative_to(ROOT)),
            "report": str(REPORT_TXT.relative_to(ROOT)),
            "completion_note": str(COMPLETE_NOTE.relative_to(ROOT)),
        },
    }

    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    with GLOBAL_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(global_summary, handle, indent=2, ensure_ascii=False)

    model_fields = [
        "global_macro_f1_rank",
        "model_id",
        "device_count",
        "mean_test_macro_f1",
        "median_test_macro_f1",
        "sample_std_test_macro_f1",
        "minimum_test_macro_f1",
        "maximum_test_macro_f1",
        "mean_test_accuracy",
        "mean_record_weighted_macro_f1",
        "mean_validation_test_macro_f1_drop",
        "maximum_validation_test_macro_f1_drop",
        "maximum_eligible_attack_fnr",
        "mean_total_seconds",
        "median_total_seconds",
        "device_macro_f1_win_count",
        "device_fastest_count",
        "local_severe_cell_count",
        "meaningful_validation_test_drop_count",
    ]
    write_csv(MODEL_SUMMARY_CSV, model_summaries, model_fields)

    matrix_fields = [
        "held_out_device",
        "model_id",
        "run_id",
        "seed",
        "test_present_class_count",
        "test_class_counts",
        "validation_macro_f1",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_record_weighted_macro_f1",
        "validation_test_macro_f1_drop",
        "benign_fnr",
        "gafgyt_fnr",
        "mirai_fnr",
        "maximum_eligible_attack_fnr",
        "local_severe_signal",
        "meaningful_validation_test_drop",
        "parameter_count",
        "completed_epoch",
        "best_epoch",
        "stopped_early",
        "completed_iterations",
        "total_seconds",
        "lock_summary_file",
        "run_result_file",
    ]
    sorted_matrix = sorted(
        model_rows,
        key=lambda row: (
            EXPECTED_DEVICES.index(row["held_out_device"]),
            EXPECTED_MODELS.index(row["model_id"]),
        ),
    )
    write_csv(MATRIX_CSV, sorted_matrix, matrix_fields)

    report_lines = [
        "=" * 86,
        "EARLY LODO GLOBAL GATE",
        "=" * 86,
        f"Devices                  : {len(summaries)}/9",
        f"Locked scientific runs   : {len(model_rows)}/27",
        f"Gate status              : {gate_status}",
        f"Local severe cells       : {len(severe_cells)}",
        (
            "Meaningful val-test drops: "
            f"{len(meaningful_drop_cells)}"
        ),
        "",
        "MODEL COMPARISON",
    ]
    for row in model_summaries:
        report_lines.append(
            "rank={rank} | {model} | mean Macro-F1={mean_f1} | "
            "min Macro-F1={min_f1} | max attack FNR={max_fnr} | "
            "mean seconds={seconds} | device wins={wins} | "
            "fastest={fastest}".format(
                rank=row["global_macro_f1_rank"],
                model=row["model_id"],
                mean_f1=fmt(row["mean_test_macro_f1"]),
                min_f1=fmt(row["minimum_test_macro_f1"]),
                max_fnr=fmt(row["maximum_eligible_attack_fnr"]),
                seconds=fmt(row["mean_total_seconds"], 3),
                wins=row["device_macro_f1_win_count"],
                fastest=row["device_fastest_count"],
            )
        )

    report_lines.extend(
        [
            "",
            "GLOBAL STATISTICS",
            f"Mean cell Macro-F1       : {fmt(mean(all_macro_values))}",
            f"Median cell Macro-F1     : {fmt(median(all_macro_values))}",
            f"Minimum cell Macro-F1    : {fmt(min(all_macro_values))}",
            f"Maximum cell Macro-F1    : {fmt(max(all_macro_values))}",
            (
                "Maximum eligible attack FNR: "
                f"{fmt(max(all_attack_fnr_values))}"
            ),
            "",
            "CLASS COVERAGE NOTE",
            (
                "Ennio and Samsung held-out tests contain no Mirai. "
                "Their Macro-F1 values use the classes present in test."
            ),
            "",
            "VALIDATION CHECKS",
        ]
    )
    report_lines.extend(
        f"{name}: {passed}" for name, passed in validation_checks.items()
    )
    report_lines.extend(
        [
            "",
            f"Summary : {GLOBAL_SUMMARY}",
            f"Models  : {MODEL_SUMMARY_CSV}",
            f"Matrix  : {MATRIX_CSV}",
            f"Report  : {REPORT_TXT}",
            f"Note    : {COMPLETE_NOTE}",
            "",
            (
                "EARLY LODO GLOBAL GATE PASSED"
                if gate_passed
                else "EARLY LODO GLOBAL GATE FINALIZED WITH FRAGILITY"
            ),
        ]
    )
    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    markdown_lines = [
        "# Early LODO Global Completion",
        "",
        f"- Generated: `{generated_at}`",
        "- Devices: `9/9`",
        "- Locked scientific runs: `27/27`",
        f"- Gate status: `{gate_status}`",
        f"- Local severe cells: `{len(severe_cells)}`",
        "",
        "## Model ranking",
        "",
        (
            "Ranking uses the unweighted mean test fingerprint Macro-F1 "
            "across the nine held-out devices."
        ),
        "",
        "| Rank | Model | Mean Macro-F1 | Minimum Macro-F1 | "
        "Max attack FNR | Mean seconds | Device wins |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summaries:
        markdown_lines.append(
            "| {rank} | {model} | {mean_f1} | {min_f1} | {max_fnr} | "
            "{seconds} | {wins} |".format(
                rank=row["global_macro_f1_rank"],
                model=row["model_id"],
                mean_f1=fmt(row["mean_test_macro_f1"]),
                min_f1=fmt(row["minimum_test_macro_f1"]),
                max_fnr=fmt(row["maximum_eligible_attack_fnr"]),
                seconds=fmt(row["mean_total_seconds"], 3),
                wins=row["device_macro_f1_win_count"],
            )
        )
    markdown_lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            (
                "Ennio and Samsung held-out tests contain no Mirai. "
                "Their Macro-F1 values are computed over the classes present. "
                "The global ranking must be read with this coverage limitation."
            ),
            "",
            "Test data were not used for model selection.",
            "",
        ]
    )
    COMPLETE_NOTE.write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )

    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
