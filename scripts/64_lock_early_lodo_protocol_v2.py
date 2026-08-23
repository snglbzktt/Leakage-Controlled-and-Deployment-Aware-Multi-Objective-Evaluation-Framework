from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
import torch
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)


PROJECT_ROOT = Path.cwd()

MEMBERSHIP_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_membership_index_summary_v2.json"
)

FOLD_FEASIBILITY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_fold_feasibility_v2.csv"
)

DEVICE_VALIDATION = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_membership_device_validation_v2.csv"
)

ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
    / "alignment.json"
)

PROTOCOL_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

RUN_MATRIX_CSV = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_run_matrix_v2.csv"
)

MANIFEST_CSV = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_release_manifest_v2.csv"
)

COMPLETION_NOTE = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "PHASE_2D_EARLY_LODO_PROTOCOL_LOCKED.md"
)


EXPECTED_DEVICE_COUNT = 9
EXPECTED_RUN_COUNT = 27
LOCKED_SEED = 2026

TWO_FAMILY_DEVICES = {
    "Ennio_Doorbell",
    "Samsung_SNH_1011_N_Webcam",
}

MODEL_CONFIGURATIONS = [
    {
        "model_id":
            "tinyml_mlp_b0",

        "model_family":
            "neural",

        "architecture":
            "tinyml_mlp",

        "hidden_layers":
            [64, 32],

        "activation":
            "relu",

        "dropout":
            0.0,

        "learning_rate":
            0.003,

        "optimizer":
            "adamw",

        "weight_decay":
            0.0001,

        "batch_size":
            4096,

        "maximum_epochs":
            15,

        "early_stopping_patience":
            4,

        "early_stopping_min_delta":
            0.0002,

        "class_weight":
            "inverse_sqrt_train_fingerprint_frequency",
    },
    {
        "model_id":
            "compact_dnn_b0",

        "model_family":
            "neural",

        "architecture":
            "compact_dnn",

        "hidden_layers":
            [128, 64, 32],

        "activation":
            "relu",

        "dropout":
            0.1,

        "learning_rate":
            0.001,

        "optimizer":
            "adamw",

        "weight_decay":
            0.0001,

        "batch_size":
            4096,

        "maximum_epochs":
            15,

        "early_stopping_patience":
            4,

        "early_stopping_min_delta":
            0.0002,

        "class_weight":
            "inverse_sqrt_train_fingerprint_frequency",
    },
    {
        "model_id":
            "hist_gradient_boosting_b0",

        "model_family":
            "classical",

        "architecture":
            "hist_gradient_boosting",

        "learning_rate":
            0.1,

        "max_iter":
            100,

        "max_leaf_nodes":
            31,

        "max_depth":
            None,

        "min_samples_leaf":
            20,

        "l2_regularization":
            0.0001,

        "max_bins":
            255,

        "class_weight":
            "balanced",

        "early_stopping":
            False,

        "loss":
            "log_loss",
    },
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Boolean value expected: {value!r}"
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


def write_csv(
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


def save_json(
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


required_files = [
    MEMBERSHIP_SUMMARY,
    FOLD_FEASIBILITY,
    DEVICE_VALIDATION,
    ALIGNMENT_FILE,
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


membership_summary = json.loads(
    MEMBERSHIP_SUMMARY.read_text(
        encoding="utf-8",
    )
)

alignment = json.loads(
    ALIGNMENT_FILE.read_text(
        encoding="utf-8",
    )
)

fold_rows = load_csv(
    FOLD_FEASIBILITY
)

device_rows = load_csv(
    DEVICE_VALIDATION
)


summary_validation = (
    membership_summary.get(
        "validation_checks",
        {},
    )
)

all_membership_checks_true = (
    membership_summary.get(
        "all_checks_passed"
    )
    is True
    and bool(summary_validation)
    and all(
        value is True
        for value
        in summary_validation.values()
    )
)


fold_devices = [
    row["held_out_device"]
    for row in fold_rows
]

device_validation_names = [
    row["device"]
    for row in device_rows
]


all_fold_overlap_zero = all(
    int(
        row[
            "fingerprint_overlap_count"
        ]
    )
    == 0
    for row in fold_rows
)

all_train_classes_present = all(
    int(row["train_benign_count"]) > 0
    and int(row["train_gafgyt_count"]) > 0
    and int(row["train_mirai_count"]) > 0
    for row in fold_rows
)

all_validation_classes_present = all(
    parse_bool(
        row[
            "validation_all_classes_present"
        ]
    )
    for row in fold_rows
)

all_tests_nonempty = all(
    int(
        row[
            "test_fingerprint_count"
        ]
    )
    > 0
    for row in fold_rows
)


test_family_policy_matches = True
mirai_absence_policy_matches = True

for row in fold_rows:
    device = row[
        "held_out_device"
    ]

    present_family_count = int(
        row[
            "test_present_family_count"
        ]
    )

    mirai_count = int(
        row[
            "test_mirai_count"
        ]
    )

    if device in TWO_FAMILY_DEVICES:
        test_family_policy_matches &= (
            present_family_count == 2
        )

        mirai_absence_policy_matches &= (
            mirai_count == 0
        )

    else:
        test_family_policy_matches &= (
            present_family_count == 3
        )

        mirai_absence_policy_matches &= (
            mirai_count > 0
        )


all_device_counts_match = all(
    parse_bool(
        row["summary_match"]
    )
    for row in device_rows
)

all_device_family_counts_match = all(
    parse_bool(
        row["family_match"]
    )
    for row in device_rows
)


hgb_signature = inspect.signature(
    HistGradientBoostingClassifier
)

hgb_parameter_names = set(
    hgb_signature.parameters
)

required_hgb_parameters = {
    "loss",
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "max_bins",
    "class_weight",
    "early_stopping",
    "random_state",
}

hgb_parameters_supported = (
    required_hgb_parameters
    .issubset(
        hgb_parameter_names
    )
)


run_matrix: list[
    dict[str, Any]
] = []

for fold_index, fold in enumerate(
    fold_rows
):
    held_out_device = fold[
        "held_out_device"
    ]

    test_family_count = int(
        fold[
            "test_present_family_count"
        ]
    )

    mirai_metric_policy = (
        "not_applicable"
        if held_out_device
        in TWO_FAMILY_DEVICES
        else "required"
    )

    for model_index, model in enumerate(
        MODEL_CONFIGURATIONS
    ):
        run_number = (
            fold_index
            * len(
                MODEL_CONFIGURATIONS
            )
            + model_index
            + 1
        )

        run_matrix.append(
            {
                "run_number":
                    run_number,

                "run_id":
                    (
                        f"early_lodo_"
                        f"{fold_index:02d}_"
                        f"{model['model_id']}_"
                        f"seed{LOCKED_SEED}"
                    ),

                "held_out_device_id":
                    fold["device_id"],

                "held_out_device":
                    held_out_device,

                "model_id":
                    model["model_id"],

                "model_family":
                    model[
                        "model_family"
                    ],

                "seed":
                    LOCKED_SEED,

                "task":
                    "family_3",

                "pipeline":
                    "pipeline_a",

                "fold_policy":
                    (
                        "strict_heldout_device_"
                        "fingerprint_disjoint"
                    ),

                "train_fingerprint_count":
                    fold[
                        "train_fingerprint_count"
                    ],

                "validation_fingerprint_count":
                    fold[
                        "validation_fingerprint_count"
                    ],

                "test_fingerprint_count":
                    fold[
                        "test_fingerprint_count"
                    ],

                "test_present_family_count":
                    test_family_count,

                "mirai_metric_policy":
                    mirai_metric_policy,

                "status":
                    "pending",
            }
        )


run_ids = [
    row["run_id"]
    for row in run_matrix
]

expected_eligible_attack_cells = (
    EXPECTED_DEVICE_COUNT
    + (
        EXPECTED_DEVICE_COUNT
        - len(
            TWO_FAMILY_DEVICES
        )
    )
)

one_third_attack_cell_threshold = (
    math.ceil(
        expected_eligible_attack_cells
        / 3
    )
)


validation_checks = {
    "membership_summary_locked":
        all_membership_checks_true,

    "nine_fold_rows":
        len(fold_rows)
        == EXPECTED_DEVICE_COUNT,

    "nine_device_validation_rows":
        len(device_rows)
        == EXPECTED_DEVICE_COUNT,

    "unique_fold_devices":
        len(set(fold_devices))
        == EXPECTED_DEVICE_COUNT,

    "fold_and_device_names_match":
        set(fold_devices)
        == set(
            device_validation_names
        ),

    "all_fold_overlap_zero":
        all_fold_overlap_zero,

    "all_train_classes_present":
        all_train_classes_present,

    "all_validation_classes_present":
        all_validation_classes_present,

    "all_tests_nonempty":
        all_tests_nonempty,

    "test_family_policy_matches":
        test_family_policy_matches,

    "mirai_absence_policy_matches":
        mirai_absence_policy_matches,

    "all_device_counts_match":
        all_device_counts_match,

    "all_device_family_counts_match":
        all_device_family_counts_match,

    "hgb_parameters_supported":
        hgb_parameters_supported,

    "three_models_registered":
        len(
            MODEL_CONFIGURATIONS
        )
        == 3,

    "twenty_seven_runs":
        len(run_matrix)
        == EXPECTED_RUN_COUNT,

    "unique_run_ids":
        len(set(run_ids))
        == EXPECTED_RUN_COUNT,

    "single_locked_seed":
        {
            int(row["seed"])
            for row in run_matrix
        }
        == {
            LOCKED_SEED
        },
}


all_checks_passed = all(
    validation_checks.values()
)


protocol = {
    "protocol_version":
        "early_lodo_protocol_v2_1",

    "status":
        (
            "locked"
            if all_checks_passed
            else "failed"
        ),

    "locked_at":
        utc_now(),

    "software": {
        "python":
            sys.version,

        "torch":
            torch.__version__,

        "scikit_learn":
            sklearn.__version__,

        "cuda_available":
            torch.cuda.is_available(),
    },

    "experiment_scope": {
        "task":
            "family_3",

        "classes": [
            "benign",
            "gafgyt",
            "mirai",
        ],

        "device_count":
            EXPECTED_DEVICE_COUNT,

        "model_count":
            len(
                MODEL_CONFIGURATIONS
            ),

        "seed_count":
            1,

        "locked_seed":
            LOCKED_SEED,

        "expected_run_count":
            EXPECTED_RUN_COUNT,
    },

    "data_policy": {
        "membership_source":
            (
                "data/cache/"
                "nbaiot_duplicate_audit.sqlite:"
                "vector_device"
            ),

        "membership_index":
            (
                "data/processed/"
                "nbaiot_early_lodo_membership_v2"
            ),

        "row_alignment":
            alignment.get(
                "row_alignment"
            ),

        "training_unit":
            "one_row_per_exact_fingerprint",

        "training_sample_weight":
            "none",

        "test_record_weight":
            (
                "held_out_device_occurrence_count"
            ),

        "pipeline":
            "pipeline_a",

        "pipeline_definition":
            (
                "source float64 -> canonical "
                "float32 -> fold-train scaler "
                "fit -> standardized float32"
            ),

        "scaler_policy":
            (
                "Fit a new StandardScaler using "
                "only the strict training rows "
                "of each held-out-device fold."
            ),

        "tree_baseline_scaling":
            (
                "HistGradientBoosting receives "
                "canonical float32 features "
                "without standardization."
            ),
    },

    "fold_policy": {
        "name":
            (
                "strict_heldout_device_"
                "fingerprint_disjoint"
            ),

        "test":
            (
                "Every fingerprint observed on "
                "the held-out device."
            ),

        "train":
            (
                "Locked train rows after removing "
                "every fingerprint observed on "
                "the held-out device."
            ),

        "validation":
            (
                "Locked validation rows after "
                "removing every fingerprint "
                "observed on the held-out device."
            ),

        "unused":
            (
                "Non-held-out rows belonging to "
                "the old locked test split are "
                "not used in the early LODO pilot."
            ),

        "fingerprint_overlap":
            "zero by construction",

        "test_not_used_for_selection":
            True,
    },

    "models":
        MODEL_CONFIGURATIONS,

    "neural_selection": {
        "selection_split":
            "strict fold validation",

        "selection_metric":
            "validation_macro_f1",

        "best_epoch_rule":
            (
                "Highest validation Macro-F1; "
                "earliest epoch breaks a tie."
            ),

        "test_evaluation":
            (
                "Exactly once after selecting "
                "the best validation epoch."
            ),
    },

    "classical_selection": {
        "configuration":
            "fixed preregistered B0",

        "validation_tuning":
            False,

        "test_tuning":
            False,
    },

    "reporting_views": {
        "primary":
            {
                "name":
                    "fingerprint_level",

                "sample_weight":
                    "none",

                "fold_macro_f1":
                    (
                        "Macro-F1 over classes "
                        "present in that test fold."
                    ),

                "device_macro_f1":
                    (
                        "Arithmetic mean of the "
                        "nine fold Macro-F1 values."
                    ),
            },

        "secondary":
            {
                "name":
                    "heldout_record_weighted",

                "sample_weight":
                    (
                        "Held-out-device occurrence "
                        "count for each fingerprint."
                    ),
            },
    },

    "metric_policy": {
        "required_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_fnr",
            "confusion_matrix",
        ],

        "absent_test_class":
            (
                "Recall, F1 and FNR are N/A; "
                "never replace with zero."
            ),

        "two_family_devices":
            sorted(
                TWO_FAMILY_DEVICES
            ),

        "mirai_eligible_fold_count":
            (
                EXPECTED_DEVICE_COUNT
                - len(
                    TWO_FAMILY_DEVICES
                )
            ),

        "gafgyt_eligible_fold_count":
            EXPECTED_DEVICE_COUNT,

        "eligible_attack_cells":
            expected_eligible_attack_cells,
    },

    "fragility_rules": {
        "primary_view":
            "fingerprint_level",

        "device_macro_f1_threshold":
            0.85,

        "median_eligible_attack_fnr_threshold":
            0.40,

        "eligible_attack_cell_fnr_threshold":
            0.40,

        "minimum_bad_attack_cells":
            one_third_attack_cell_threshold,

        "minimum_bad_attack_cell_fraction":
            "one_third",

        "single_bad_cell_interpretation":
            (
                "Local severe failure only; "
                "not sufficient alone for the "
                "global fragility conclusion."
            ),

        "meaningful_closed_set_macro_f1_drop":
            0.10,

        "meaningful_attack_fnr_increase":
            0.10,
    },

    "reproducibility": {
        "seed":
            LOCKED_SEED,

        "deterministic_torch":
            True,

        "no_hyperparameter_search":
            True,

        "run_matrix_file":
            str(RUN_MATRIX_CSV),
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json(
    PROTOCOL_JSON,
    protocol,
)


write_csv(
    RUN_MATRIX_CSV,
    run_matrix,
    list(
        run_matrix[0].keys()
    ),
)


completion_text = f"""# Faz 2D — Erken LODO Protokolü Kilitlendi

Kilitleme zamanı: `{protocol["locked_at"]}`

## Deney kapsamı

- Görev: `family_3`
- Cihaz sayısı: `{EXPECTED_DEVICE_COUNT}`
- Model sayısı: `3`
- Seed: `{LOCKED_SEED}`
- Toplam çalışma: `{EXPECTED_RUN_COUNT}`

## Modeller

1. `tinyml_mlp_b0`
2. `compact_dnn_b0`
3. `hist_gradient_boosting_b0`

## Katlama politikası

Test kümesi, tutulan cihazda görülen bütün exact parmak
izlerini içerir. Bu parmak izlerinin tamamı eğitim ve doğrulama
kümelerinden çıkarılır. Bu nedenle eğitim–test exact fingerprint
örtüşmesi sıfırdır.

## Ana değerlendirme

Ana değerlendirme parmak izi seviyesindedir. İkincil değerlendirme,
tutulan cihazın occurrence sayıları kullanılarak kayıt ağırlıklı
olarak yapılır.

Ennio_Doorbell ve Samsung_SNH_1011_N_Webcam katlarında Mirai sınıfı
bulunmadığı için Mirai Recall, F1 ve FNR değerleri `N/A` raporlanır.

## Kırılganlık kuralları

- Device-macro Macro-F1 `< 0.85`
- Uygun saldırı hücrelerinin medyan FNR değeri `> 0.40`
- En az `{one_third_attack_cell_threshold}` uygun cihaz-saldırı
  hücresinde FNR `> 0.40`

Tek bir yüksek FNR hücresi yerel ciddi başarısızlık olarak raporlanır;
tek başına genel kırılganlık sonucuna dönüştürülmez.

## Dosyalar

- `{PROTOCOL_JSON.relative_to(PROJECT_ROOT)}`
- `{RUN_MATRIX_CSV.relative_to(PROJECT_ROOT)}`
- `{MANIFEST_CSV.relative_to(PROJECT_ROOT)}`
"""


COMPLETION_NOTE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_NOTE.write_text(
    completion_text,
    encoding="utf-8",
)


manifest_paths = [
    MEMBERSHIP_SUMMARY,
    FOLD_FEASIBILITY,
    DEVICE_VALIDATION,
    ALIGNMENT_FILE,
    PROTOCOL_JSON,
    RUN_MATRIX_CSV,
    COMPLETION_NOTE,
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


write_csv(
    MANIFEST_CSV,
    manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


print("=" * 78)
print("PHASE 2D EARLY LODO PROTOCOL LOCK")
print("=" * 78)
print(
    f"Devices           : "
    f"{len(fold_rows)}"
)
print(
    f"Models            : "
    f"{len(MODEL_CONFIGURATIONS)}"
)
print(
    f"Seed              : "
    f"{LOCKED_SEED}"
)
print(
    f"Expected runs     : "
    f"{EXPECTED_RUN_COUNT}"
)
print(
    f"Generated runs    : "
    f"{len(run_matrix)}"
)
print(
    f"Eligible attacks  : "
    f"{expected_eligible_attack_cells}"
)
print(
    f"One-third cutoff  : "
    f"{one_third_attack_cell_threshold}"
)

print()
print("MODELS")

for model in MODEL_CONFIGURATIONS:
    print(
        f"- {model['model_id']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Protocol : {PROTOCOL_JSON}")
print(f"Runs     : {RUN_MATRIX_CSV}")
print(f"Manifest : {MANIFEST_CSV}")
print(f"Note     : {COMPLETION_NOTE}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2D EARLY LODO "
        "PROTOCOL LOCK FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2D EARLY LODO "
    "PROTOCOL LOCKED"
)
