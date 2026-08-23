from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path.cwd()

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

RUN_MATRIX_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_run_matrix_v2.csv"
)

RUNNER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "72_run_early_lodo_neural_experiment_v2.py"
)

RUN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "runs"
)

LOCK_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "locks"
)

LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_execution_ledger_v2.csv"
)

DOC_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "early_lodo"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

LOCAL_MACRO_F1_THRESHOLD = 0.85
LOCAL_ATTACK_FNR_THRESHOLD = 0.40
TOLERANCE = 1e-10


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_name(
    value: str,
) -> str:
    result = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return result.strip("_")


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def save_json_atomic(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
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
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def finite_unit_interval(
    value: Any,
) -> bool:
    if value is None:
        return False

    number = float(value)

    return (
        math.isfinite(number)
        and 0.0 <= number <= 1.0
    )


def optional_finite_unit_interval(
    value: Any,
) -> bool:
    if value is None:
        return True

    return finite_unit_interval(
        value
    )


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-id",
    required=True,
)

args = parser.parse_args()

run_id = str(
    args.run_id
)


run_directory = (
    RUN_ROOT
    / run_id
)

result_file = (
    run_directory
    / "run_result.json"
)

history_file = (
    run_directory
    / "history.csv"
)

best_model_file = (
    run_directory
    / "best_model.pt"
)

checkpoint_file = (
    run_directory
    / "checkpoint.pt"
)

report_file = (
    run_directory
    / "run_report.txt"
)


required_files = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    RUNNER_SCRIPT,
    result_file,
    history_file,
    best_model_file,
    checkpoint_file,
    report_file,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


protocol = load_json(
    PROTOCOL_FILE
)

run_matrix = load_csv(
    RUN_MATRIX_FILE
)

result = load_json(
    result_file
)

history_rows = load_csv(
    history_file
)


matching_run_rows = [
    row
    for row in run_matrix
    if row["run_id"] == run_id
]

if len(matching_run_rows) != 1:
    raise RuntimeError(
        "Run matrix identity was not "
        "found exactly once."
    )

run_spec = matching_run_rows[0]

held_out_device = (
    run_spec["held_out_device"]
)

model_id = run_spec["model_id"]

seed = int(
    run_spec["seed"]
)


cache_directory = (
    CACHE_ROOT
    / safe_name(
        held_out_device
    )
)

cache_summary_file = (
    cache_directory
    / "cache_summary.json"
)

if not cache_summary_file.exists():
    raise RuntimeError(
        "Validated fold cache summary "
        "was not found."
    )

cache_summary = load_json(
    cache_summary_file
)


lock_directory = (
    LOCK_ROOT
    / run_id
)

lock_summary_file = (
    lock_directory
    / "lock_summary.json"
)

manifest_file = (
    lock_directory
    / "release_manifest.csv"
)

completion_note = (
    DOC_ROOT
    / (
        run_id
        + "_LOCKED.md"
    )
)


result_run = result.get(
    "run",
    {},
)

result_training = result.get(
    "training",
    {},
)

result_test = result.get(
    "test",
    {},
)

result_data = result.get(
    "data",
    {},
)

fingerprint_metrics = (
    result_test.get(
        "fingerprint_level",
        {},
    )
)

record_metrics = (
    result_test.get(
        "heldout_record_weighted",
        {},
    )
)

per_class = (
    fingerprint_metrics.get(
        "per_class",
        {},
    )
)

result_checks = result.get(
    "validation_checks",
    {},
)


completed_epoch = int(
    result_training.get(
        "completed_epoch",
        -1,
    )
)

best_epoch = int(
    result_training.get(
        "best_epoch",
        -1,
    )
)

best_validation_macro_f1 = float(
    result_training.get(
        "best_validation_macro_f1",
        float("nan"),
    )
)

stopped_early = bool(
    result_training.get(
        "stopped_early",
        False,
    )
)


history_epochs = [
    int(row["epoch"])
    for row in history_rows
]

history_validation_f1 = [
    float(
        row[
            "validation_macro_f1"
        ]
    )
    for row in history_rows
]

history_bad_counts = [
    int(
        row["bad_epoch_count"]
    )
    for row in history_rows
]

history_new_best = [
    parse_bool(
        row["is_new_best"]
    )
    for row in history_rows
]


if history_validation_f1:
    history_max_f1 = max(
        history_validation_f1
    )

    history_best_epoch = min(
        epoch
        for epoch, metric
        in zip(
            history_epochs,
            history_validation_f1,
        )
        if abs(
            metric - history_max_f1
        )
        <= TOLERANCE
    )

else:
    history_max_f1 = float("nan")
    history_best_epoch = -1


protocol_sha256 = sha256_file(
    PROTOCOL_FILE
)

run_matrix_sha256 = sha256_file(
    RUN_MATRIX_FILE
)

cache_summary_sha256 = sha256_file(
    cache_summary_file
)

result_identity = result.get(
    "identity",
    {},
)


best_model_checkpoint = torch.load(
    best_model_file,
    map_location="cpu",
    weights_only=False,
)

resume_checkpoint = torch.load(
    checkpoint_file,
    map_location="cpu",
    weights_only=False,
)


fingerprint_confusion = np.asarray(
    fingerprint_metrics.get(
        "confusion_matrix",
        [],
    ),
    dtype=np.float64,
)

record_confusion = np.asarray(
    record_metrics.get(
        "confusion_matrix",
        [],
    ),
    dtype=np.float64,
)

test_class_counts = np.asarray(
    result_data.get(
        "test_class_counts",
        [],
    ),
    dtype=np.float64,
)


fingerprint_macro_f1 = float(
    fingerprint_metrics.get(
        "macro_f1_present_classes",
        float("nan"),
    )
)

fingerprint_accuracy = float(
    fingerprint_metrics.get(
        "accuracy",
        float("nan"),
    )
)

record_macro_f1 = float(
    record_metrics.get(
        "macro_f1_present_classes",
        float("nan"),
    )
)


gafgyt_fnr = (
    per_class.get(
        "gafgyt",
        {},
    ).get("fnr")
)

mirai_fnr = (
    per_class.get(
        "mirai",
        {},
    ).get("fnr")
)

benign_fnr = (
    per_class.get(
        "benign",
        {},
    ).get("fnr")
)


local_macro_f1_signal = (
    fingerprint_macro_f1
    < LOCAL_MACRO_F1_THRESHOLD
)

local_gafgyt_fnr_signal = (
    gafgyt_fnr is not None
    and float(gafgyt_fnr)
    > LOCAL_ATTACK_FNR_THRESHOLD
)

local_mirai_fnr_signal = (
    mirai_fnr is not None
    and float(mirai_fnr)
    > LOCAL_ATTACK_FNR_THRESHOLD
)

local_severe_signal = bool(
    local_macro_f1_signal
    or local_gafgyt_fnr_signal
    or local_mirai_fnr_signal
)


maximum_epochs = int(
    result.get(
        "model",
        {},
    ).get(
        "maximum_epochs",
        -1,
    )
)

patience = int(
    result.get(
        "model",
        {},
    ).get(
        "early_stopping_patience",
        -1,
    )
)

final_bad_epoch_count = (
    history_bad_counts[-1]
    if history_bad_counts
    else -1
)


expected_result_identity = {
    "protocol_sha256":
        protocol_sha256,

    "run_matrix_sha256":
        run_matrix_sha256,

    "cache_summary_sha256":
        cache_summary_sha256,
}


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked"
        and protocol.get(
            "all_checks_passed"
        )
        is True,

    "run_result_completed":
        result.get("status")
        == "completed",

    "scientific_result_true":
        result.get(
            "scientific_result"
        )
        is True,

    "result_all_checks_passed":
        result.get(
            "all_checks_passed"
        )
        is True,

    "all_result_internal_checks_true":
        bool(result_checks)
        and all(
            value is True
            for value
            in result_checks.values()
        ),

    "run_id_matches":
        result_run.get("run_id")
        == run_id,

    "held_out_device_matches":
        result_run.get(
            "held_out_device"
        )
        == held_out_device,

    "model_id_matches":
        result_run.get(
            "model_id"
        )
        == model_id,

    "seed_matches":
        int(
            result_run.get(
                "seed",
                -1,
            )
        )
        == seed,

    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True
        and cache_summary.get(
            "held_out_device"
        )
        == held_out_device,

    "identity_hashes_match":
        result_identity
        == expected_result_identity,

    "history_nonempty":
        len(history_rows) > 0,

    "history_row_count_matches_completed_epoch":
        len(history_rows)
        == completed_epoch,

    "history_epochs_are_sequential":
        history_epochs
        == list(
            range(
                1,
                completed_epoch + 1,
            )
        ),

    "best_epoch_matches_history":
        best_epoch
        == history_best_epoch,

    "best_metric_matches_history":
        math.isclose(
            best_validation_macro_f1,
            history_max_f1,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "best_epoch_marked_new_best":
        (
            1 <= best_epoch
            <= len(history_new_best)
            and history_new_best[
                best_epoch - 1
            ]
        ),

    "completed_epoch_within_budget":
        1
        <= completed_epoch
        <= maximum_epochs,

    "early_stopping_consistent":
        (
            stopped_early
            and final_bad_epoch_count
            >= patience
        )
        or (
            not stopped_early
            and completed_epoch
            == maximum_epochs
        ),

    "test_evaluated_once":
        int(
            result_test.get(
                "evaluation_count",
                -1,
            )
        )
        == 1,

    "fingerprint_metrics_finite":
        finite_unit_interval(
            fingerprint_macro_f1
        )
        and finite_unit_interval(
            fingerprint_accuracy
        )
        and finite_unit_interval(
            fingerprint_metrics.get(
                "balanced_accuracy"
            )
        )
        and finite_unit_interval(
            fingerprint_metrics.get(
                "weighted_f1"
            )
        ),

    "record_metrics_finite":
        finite_unit_interval(
            record_macro_f1
        )
        and finite_unit_interval(
            record_metrics.get(
                "accuracy"
            )
        )
        and finite_unit_interval(
            record_metrics.get(
                "balanced_accuracy"
            )
        )
        and finite_unit_interval(
            record_metrics.get(
                "weighted_f1"
            )
        ),

    "per_class_metrics_valid":
        all(
            class_name in per_class
            and optional_finite_unit_interval(
                per_class[
                    class_name
                ].get("precision")
            )
            and optional_finite_unit_interval(
                per_class[
                    class_name
                ].get("recall")
            )
            and optional_finite_unit_interval(
                per_class[
                    class_name
                ].get("f1")
            )
            and optional_finite_unit_interval(
                per_class[
                    class_name
                ].get("fnr")
            )
            for class_name
            in CLASS_NAMES
        ),

    "fingerprint_confusion_shape":
        fingerprint_confusion.shape
        == (
            3,
            3,
        ),

    "record_confusion_shape":
        record_confusion.shape
        == (
            3,
            3,
        ),

    "fingerprint_supports_match":
        fingerprint_confusion.shape
        == (
            3,
            3,
        )
        and test_class_counts.shape
        == (
            3,
        )
        and np.allclose(
            fingerprint_confusion.sum(
                axis=1
            ),
            test_class_counts,
            rtol=0.0,
            atol=TOLERANCE,
        ),

    "fingerprint_total_matches":
        fingerprint_confusion.shape
        == (
            3,
            3,
        )
        and math.isclose(
            float(
                fingerprint_confusion.sum()
            ),
            float(
                result_data.get(
                    "test_fingerprint_count",
                    -1,
                )
            ),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "record_weight_total_matches":
        record_confusion.shape
        == (
            3,
            3,
        )
        and math.isclose(
            float(
                record_confusion.sum()
            ),
            float(
                cache_summary.get(
                    "observed_test_occurrence_count",
                    -1,
                )
            ),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "best_model_identity_matches":
        best_model_checkpoint.get(
            "run_id"
        )
        == run_id
        and best_model_checkpoint.get(
            "held_out_device"
        )
        == held_out_device
        and best_model_checkpoint.get(
            "model_id"
        )
        == model_id
        and int(
            best_model_checkpoint.get(
                "seed",
                -1,
            )
        )
        == seed,

    "best_model_epoch_matches":
        int(
            best_model_checkpoint.get(
                "best_epoch",
                -1,
            )
        )
        == best_epoch
        and math.isclose(
            float(
                best_model_checkpoint.get(
                    "best_validation_macro_f1",
                    float("nan"),
                )
            ),
            best_validation_macro_f1,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "checkpoint_identity_matches":
        resume_checkpoint.get(
            "identity"
        )
        == {
            "run_id":
                run_id,

            **expected_result_identity,
        },

    "checkpoint_completion_matches":
        int(
            resume_checkpoint.get(
                "completed_epoch",
                -1,
            )
        )
        == completed_epoch,

    "checkpoint_stop_state_matches":
        bool(
            resume_checkpoint.get(
                "should_stop",
                False,
            )
        )
        == stopped_early,

    "all_output_files_exist":
        all(
            path.exists()
            for path in (
                result_file,
                history_file,
                best_model_file,
                checkpoint_file,
                report_file,
            )
        ),
}


all_checks_passed = all(
    validation_checks.values()
)

locked_at = utc_now()


lock_summary = {
    "protocol_version":
        "early_lodo_scientific_run_lock_v2_1",

    "status":
        (
            "locked"
            if all_checks_passed
            else "failed"
        ),

    "locked_at":
        locked_at,

    "run": {
        "run_id":
            run_id,

        "held_out_device":
            held_out_device,

        "model_id":
            model_id,

        "seed":
            seed,

        "completed_epoch":
            completed_epoch,

        "best_epoch":
            best_epoch,

        "stopped_early":
            stopped_early,
    },

    "metrics": {
        "best_validation_macro_f1":
            best_validation_macro_f1,

        "test_fingerprint_macro_f1":
            fingerprint_macro_f1,

        "test_fingerprint_accuracy":
            fingerprint_accuracy,

        "test_record_weighted_macro_f1":
            record_macro_f1,

        "benign_fnr":
            benign_fnr,

        "gafgyt_fnr":
            gafgyt_fnr,

        "mirai_fnr":
            mirai_fnr,
    },

    "local_gate_assessment": {
        "macro_f1_threshold":
            LOCAL_MACRO_F1_THRESHOLD,

        "attack_fnr_threshold":
            LOCAL_ATTACK_FNR_THRESHOLD,

        "macro_f1_below_threshold":
            local_macro_f1_signal,

        "gafgyt_fnr_above_threshold":
            local_gafgyt_fnr_signal,

        "mirai_fnr_above_threshold":
            local_mirai_fnr_signal,

        "local_severe_signal":
            local_severe_signal,

        "interpretation": (
            "No local severe fragility signal "
            "was detected for this device-model "
            "cell."
            if not local_severe_signal
            else (
                "A local severe fragility signal "
                "was detected for this device-model "
                "cell."
            )
        ),

        "overall_gate_status": (
            "pending_remaining_device_model_runs"
        ),
    },

    "identity": {
        "protocol_sha256":
            protocol_sha256,

        "run_matrix_sha256":
            run_matrix_sha256,

        "cache_summary_sha256":
            cache_summary_sha256,

        "run_result_sha256":
            sha256_file(
                result_file
            ),

        "history_sha256":
            sha256_file(
                history_file
            ),

        "best_model_sha256":
            sha256_file(
                best_model_file
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_file
            ),
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json_atomic(
    lock_summary_file,
    lock_summary,
)


if all_checks_passed:
    completion_note.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_test_drop = (
        best_validation_macro_f1
        - fingerprint_macro_f1
    )

    completion_text = f"""# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `{run_id}`
- Kilit zamanı: `{locked_at}`
- Held-out cihaz: `{held_out_device}`
- Model: `{model_id}`
- Seed: `{seed}`
- Tamamlanan epoch: `{completed_epoch}`
- Seçilen epoch: `{best_epoch}`
- Erken durdurma: `{stopped_early}`

## Sonuçlar

- Validation Macro-F1: `{best_validation_macro_f1:.9f}`
- Test fingerprint Macro-F1: `{fingerprint_macro_f1:.9f}`
- Validation-test farkı: `{validation_test_drop:.9f}`
- Test accuracy: `{fingerprint_accuracy:.9f}`
- Record-weighted Macro-F1: `{record_macro_f1:.9f}`
- Benign FNR: `{benign_fnr}`
- Gafgyt FNR: `{gafgyt_fnr}`
- Mirai FNR: `{mirai_fnr}`

## Yerel karar

- Macro-F1 < 0.85: `{local_macro_f1_signal}`
- Gafgyt FNR > 0.40: `{local_gafgyt_fnr_signal}`
- Mirai FNR > 0.40: `{local_mirai_fnr_signal}`
- Yerel ciddi kırılganlık sinyali: `{local_severe_signal}`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `{result_file.relative_to(PROJECT_ROOT)}`
- `{history_file.relative_to(PROJECT_ROOT)}`
- `{best_model_file.relative_to(PROJECT_ROOT)}`
- `{checkpoint_file.relative_to(PROJECT_ROOT)}`
- `{report_file.relative_to(PROJECT_ROOT)}`
- `{lock_summary_file.relative_to(PROJECT_ROOT)}`
- `{manifest_file.relative_to(PROJECT_ROOT)}`
"""

    completion_note.write_text(
        completion_text,
        encoding="utf-8",
    )


    manifest_paths = [
        RUNNER_SCRIPT,
        PROTOCOL_FILE,
        RUN_MATRIX_FILE,
        cache_summary_file,
        result_file,
        history_file,
        best_model_file,
        checkpoint_file,
        report_file,
        lock_summary_file,
        completion_note,
    ]

    manifest_rows = [
        {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
        for path in manifest_paths
    ]

    write_csv_atomic(
        manifest_file,
        manifest_rows,
        [
            "relative_path",
            "size_bytes",
            "sha256",
        ],
    )


    ledger_fieldnames = [
        "run_id",
        "status",
        "locked_at",
        "held_out_device",
        "model_id",
        "seed",
        "completed_epoch",
        "best_epoch",
        "stopped_early",
        "best_validation_macro_f1",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_record_weighted_macro_f1",
        "benign_fnr",
        "gafgyt_fnr",
        "mirai_fnr",
        "local_severe_signal",
        "result_file",
        "lock_summary_file",
    ]

    ledger_rows = (
        load_csv(LEDGER_FILE)
        if LEDGER_FILE.exists()
        else []
    )

    ledger_rows = [
        row
        for row in ledger_rows
        if row["run_id"] != run_id
    ]

    ledger_rows.append(
        {
            "run_id":
                run_id,

            "status":
                "locked",

            "locked_at":
                locked_at,

            "held_out_device":
                held_out_device,

            "model_id":
                model_id,

            "seed":
                seed,

            "completed_epoch":
                completed_epoch,

            "best_epoch":
                best_epoch,

            "stopped_early":
                stopped_early,

            "best_validation_macro_f1":
                best_validation_macro_f1,

            "test_fingerprint_macro_f1":
                fingerprint_macro_f1,

            "test_fingerprint_accuracy":
                fingerprint_accuracy,

            "test_record_weighted_macro_f1":
                record_macro_f1,

            "benign_fnr":
                benign_fnr,

            "gafgyt_fnr":
                gafgyt_fnr,

            "mirai_fnr":
                mirai_fnr,

            "local_severe_signal":
                local_severe_signal,

            "result_file":
                str(
                    result_file.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "lock_summary_file":
                str(
                    lock_summary_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
        }
    )

    ledger_rows.sort(
        key=lambda row: row["run_id"]
    )

    write_csv_atomic(
        LEDGER_FILE,
        ledger_rows,
        ledger_fieldnames,
    )


print("=" * 86)
print("EARLY LODO SCIENTIFIC RUN LOCK")
print("=" * 86)
print(f"Run id          : {run_id}")
print(f"Held-out device : {held_out_device}")
print(f"Model           : {model_id}")
print(f"Best epoch      : {best_epoch}")
print(
    f"Validation F1   : "
    f"{best_validation_macro_f1:.9f}"
)
print(
    f"Test Macro-F1   : "
    f"{fingerprint_macro_f1:.9f}"
)
print(
    f"Gafgyt FNR      : "
    f"{gafgyt_fnr}"
)
print(
    f"Mirai FNR       : "
    f"{mirai_fnr}"
)
print(
    f"Local severe    : "
    f"{local_severe_signal}"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Lock summary : {lock_summary_file}")

if all_checks_passed:
    print(f"Manifest     : {manifest_file}")
    print(f"Ledger       : {LEDGER_FILE}")
    print(f"Note         : {completion_note}")

if not all_checks_passed:
    print()
    print(
        "EARLY LODO SCIENTIFIC "
        "RUN LOCK FAILED"
    )
    sys.exit(1)

print()
print(
    "EARLY LODO SCIENTIFIC "
    "RUN LOCKED"
)
