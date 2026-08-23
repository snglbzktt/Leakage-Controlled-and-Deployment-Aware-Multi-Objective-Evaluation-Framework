from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

SOURCE_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

MODEL_REGISTRY = (
    PROJECT_ROOT
    / "models"
    / "architecture"
    / "nbaiot_model_registry_v1.json"
)

PHASE_1D_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "numeric_pipeline_order_summary_v2.json"
)

OUTPUT_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_numeric_order_training_pilot_v2.json"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "numeric_order_training_pilot_protocol_manifest_v2.json"
)

OUTPUT_DOCUMENT = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "PHASE_1E_NUMERIC_ORDER_PILOT_PROTOCOL_LOCKED.md"
)

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "51_lock_numeric_order_training_pilot_protocol_v2.py"
)


EXPECTED_MODELS = {
    "tinyml_mlp": {
        "display_name": "TinyML-MLP",
        "parameter_count": 9_603,
        "learning_rate": 0.003,
    },
    "compact_dnn": {
        "display_name": "Compact-DNN",
        "parameter_count": 25_283,
        "learning_rate": 0.001,
    },
}

EXPECTED_MAIN_SEEDS = {
    42,
    123,
    2026,
    3407,
    8192,
}

PILOT_SEEDS = [
    42,
    123,
]


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def find_values(
    value: Any,
    key_name: str,
) -> list[Any]:
    results: list[Any] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == key_name:
                results.append(child)

            results.extend(
                find_values(
                    child,
                    key_name,
                )
            )

    elif isinstance(value, list):
        for child in value:
            results.extend(
                find_values(
                    child,
                    key_name,
                )
            )

    return results


def find_model_learning_rates(
    value: Any,
    model_name: str,
) -> list[float]:
    rates: list[float] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if (
                str(key) == model_name
                and isinstance(child, dict)
                and "learning_rate" in child
            ):
                rates.append(
                    float(
                        child["learning_rate"]
                    )
                )

            rates.extend(
                find_model_learning_rates(
                    child,
                    model_name,
                )
            )

    elif isinstance(value, list):
        for child in value:
            rates.extend(
                find_model_learning_rates(
                    child,
                    model_name,
                )
            )

    return rates


def contains_number(
    values: list[Any],
    expected: float,
    tolerance: float = 1e-12,
) -> bool:
    for value in values:
        try:
            numeric_value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isclose(
            numeric_value,
            expected,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return True

    return False


required_files = [
    SOURCE_PROTOCOL,
    MODEL_REGISTRY,
    PHASE_1D_SUMMARY,
    SCRIPT_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Eksik protokol girdileri:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


source_protocol = json.loads(
    SOURCE_PROTOCOL.read_text(
        encoding="utf-8",
    )
)

model_registry = json.loads(
    MODEL_REGISTRY.read_text(
        encoding="utf-8",
    )
)

phase_1d = json.loads(
    PHASE_1D_SUMMARY.read_text(
        encoding="utf-8",
    )
)


optimizer_values = find_values(
    source_protocol,
    "optimizer",
)

batch_size_values = find_values(
    source_protocol,
    "batch_size",
)

max_epoch_values = find_values(
    source_protocol,
    "max_epochs",
)

patience_values = find_values(
    source_protocol,
    "early_stopping_patience",
)

min_delta_values = find_values(
    source_protocol,
    "early_stopping_min_delta",
)

weight_decay_values = find_values(
    source_protocol,
    "weight_decay",
)

training_seed_values = find_values(
    source_protocol,
    "training_seeds",
)


flattened_training_seeds: set[int] = set()

for value in training_seed_values:
    if isinstance(value, list):
        for seed in value:
            try:
                flattened_training_seeds.add(
                    int(seed)
                )
            except (
                TypeError,
                ValueError,
            ):
                pass


registry_text = json.dumps(
    model_registry,
    ensure_ascii=False,
)


validation_checks = {
    "phase_1d_completed":
        phase_1d.get("status")
        == "completed",

    "phase_1d_checks_passed":
        phase_1d.get(
            "all_checks_passed"
        )
        is True,

    "phase_1d_pilot_required":
        phase_1d.get(
            "requires_training_pilot"
        )
        is True,

    "optimizer_adamw":
        any(
            str(value).casefold()
            == "adamw"
            for value in optimizer_values
        ),

    "batch_size_4096":
        contains_number(
            batch_size_values,
            4096,
        ),

    "max_epochs_15":
        contains_number(
            max_epoch_values,
            15,
        ),

    "patience_4":
        contains_number(
            patience_values,
            4,
        ),

    "minimum_delta_0_0002":
        contains_number(
            min_delta_values,
            0.0002,
        ),

    "weight_decay_0_0001":
        contains_number(
            weight_decay_values,
            0.0001,
        ),

    "main_seed_set_available":
        EXPECTED_MAIN_SEEDS.issubset(
            flattened_training_seeds
        ),

    "pilot_seeds_available":
        set(PILOT_SEEDS).issubset(
            flattened_training_seeds
        ),

    "tinyml_mlp_registered":
        "tinyml_mlp"
        in registry_text,

    "compact_dnn_registered":
        "compact_dnn"
        in registry_text,

    "tinyml_learning_rate":
        contains_number(
            find_model_learning_rates(
                source_protocol,
                "tinyml_mlp",
            ),
            0.003,
        ),

    "compact_learning_rate":
        contains_number(
            find_model_learning_rates(
                source_protocol,
                "compact_dnn",
            ),
            0.001,
        ),
}


failed_checks = [
    name
    for name, passed
    in validation_checks.items()
    if not passed
]

if failed_checks:
    print("Pilot protokolü kilitlenemedi.")
    print("Başarısız kontroller:")

    for check in failed_checks:
        print(f"- {check}")

    sys.exit(1)


run_matrix = []

for numeric_pipeline in (
    "pipeline_a",
    "pipeline_b",
):
    for model_name in (
        "tinyml_mlp",
        "compact_dnn",
    ):
        for seed in PILOT_SEEDS:
            run_matrix.append(
                {
                    "run_id":
                        (
                            f"{numeric_pipeline}__"
                            f"{model_name}__"
                            f"seed{seed}"
                        ),

                    "numeric_pipeline":
                        numeric_pipeline,

                    "model_name":
                        model_name,

                    "seed":
                        seed,

                    "learning_rate":
                        EXPECTED_MODELS[
                            model_name
                        ][
                            "learning_rate"
                        ],
                }
            )


pilot_protocol = {
    "protocol_name":
        "N-BaIoT numeric processing order training pilot",

    "protocol_version":
        "numeric_order_training_pilot_v2_1",

    "status":
        "locked_before_training",

    "locked_at":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "objective":
        (
            "Determine whether float32 conversion before "
            "standardization or after standardization produces "
            "a meaningful and repeatable change in family-3 "
            "classification performance."
        ),

    "selection_scope":
        (
            "This pilot selects only the numeric processing "
            "order. It is not a final model-comparison study."
        ),

    "dataset": {
        "name":
            "nbaiot_primary_seed2026",

        "source_precision":
            "float64",

        "split_seed":
            2026,

        "split_assignment":
            "existing locked primary splits",

        "train_rows":
            1_738_133,

        "validation_rows":
            371_884,

        "test_rows":
            372_659,

        "feature_count":
            115,
    },

    "task": {
        "name":
            "family_3",

        "classes": [
            "benign",
            "gafgyt",
            "mirai",
        ],

        "label_mapping": {
            "benign":
                "benign",

            "gafgyt_*":
                "gafgyt",

            "mirai_*":
                "mirai",
        },
    },

    "numeric_pipelines": {
        "pipeline_a": {
            "description":
                (
                    "source float64 -> float32 -> promote to "
                    "float64 for train-only StandardScaler -> "
                    "final float32 model input"
                ),

            "scaler_fit_split":
                "train_only",

            "scaler_parameter_dtype":
                "float64",

            "model_input_dtype":
                "float32",
        },

        "pipeline_b": {
            "description":
                (
                    "source float64 -> train-only StandardScaler "
                    "in float64 -> final float32 model input"
                ),

            "scaler_fit_split":
                "train_only",

            "scaler_parameter_dtype":
                "float64",

            "model_input_dtype":
                "float32",
        },
    },

    "models": EXPECTED_MODELS,

    "training_seeds":
        PILOT_SEEDS,

    "seed_selection_rationale":
        (
            "Seeds 42 and 123 are the first two seeds in the "
            "locked five-seed baseline protocol. Seed 42 is the "
            "historical tuning seed; seed 123 provides an "
            "independent matched replication."
        ),

    "run_count":
        len(run_matrix),

    "run_matrix":
        run_matrix,

    "training": {
        "optimizer":
            "AdamW",

        "batch_size":
            4096,

        "max_epochs":
            15,

        "early_stopping_patience":
            4,

        "early_stopping_min_delta":
            0.0002,

        "weight_decay":
            0.0001,

        "loss":
            "CrossEntropyLoss",

        "class_weight_scheme":
            (
                "inverse square root frequency normalized "
                "to mean 1"
            ),

        "class_weights_fitted_from":
            "primary train split only",

        "class_weights_shared_between_numeric_pipelines":
            True,

        "num_workers":
            0,

        "device":
            "cpu",

        "best_checkpoint_metric":
            "validation_macro_f1",

        "test_evaluations_per_run":
            1,
    },

    "fairness_controls": {
        "same_source_rows":
            True,

        "same_split_assignments":
            True,

        "same_label_mapping":
            True,

        "same_class_weights":
            True,

        "same_model_initialization_for_matched_seed":
            True,

        "same_training_order_for_matched_seed":
            True,

        "same_optimizer_and_stopping_rules":
            True,

        "only_intended_difference":
            "numeric processing order",
    },

    "primary_analysis": {
        "unit":
            "matched model-seed pair",

        "pair_count":
            4,

        "metric":
            "best validation Macro-F1",

        "delta_direction":
            "pipeline_b_minus_pipeline_a",

        "practical_equivalence_margin":
            0.001,

        "pipeline_b_selection_rule":
            (
                "Select pipeline B only when the median paired "
                "validation Macro-F1 delta is at least +0.001 "
                "and pipeline B wins at least 3 of 4 matched pairs."
            ),

        "pipeline_a_selection_rule":
            (
                "Select pipeline A when the median paired "
                "validation Macro-F1 delta is at most -0.001 "
                "and pipeline A wins at least 3 of 4 matched pairs."
            ),

        "inconclusive_rule":
            (
                "Otherwise classify the two orders as practically "
                "equivalent for this pilot and retain pipeline A "
                "for continuity with the current model-facing "
                "pipeline and because it produced lower split "
                "overlap in the no-training audit."
            ),

        "test_set_policy":
            (
                "Test metrics are recorded once from each "
                "validation-selected checkpoint but are not used "
                "to choose the numeric processing order."
            ),
    },

    "reporting": {
        "required_validation_metrics": [
            "loss",
            "accuracy",
            "macro_f1",
            "per_class_recall",
            "per_class_fnr",
            "confusion_matrix",
        ],

        "required_test_metrics": [
            "loss",
            "accuracy",
            "macro_f1",
            "per_class_recall",
            "per_class_fnr",
            "confusion_matrix",
        ],

        "required_runtime_fields": [
            "training_seconds",
            "best_epoch",
            "epochs_completed",
            "peak_process_memory_mb",
        ],
    },

    "source_artifacts": {
        "baseline_protocol":
            str(
                SOURCE_PROTOCOL.relative_to(
                    PROJECT_ROOT
                )
            ),

        "model_registry":
            str(
                MODEL_REGISTRY.relative_to(
                    PROJECT_ROOT
                )
            ),

        "phase_1d_summary":
            str(
                PHASE_1D_SUMMARY.relative_to(
                    PROJECT_ROOT
                )
            ),
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        True,
}


OUTPUT_PROTOCOL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PROTOCOL.write_text(
    json.dumps(
        pilot_protocol,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


document_text = f"""# Phase 1E — Numeric-Order Training Pilot Protocol Locked

- Protocol version: numeric_order_training_pilot_v2_1
- Locked before training: yes
- Models: TinyML-MLP and Compact-DNN
- Numeric pipelines: A and B
- Seeds: 42 and 123
- Total runs: 8
- Task: family-3
- Device: CPU
- Primary selection metric: validation Macro-F1
- Test used for selection: no

## Run matrix

"""

for run in run_matrix:
    document_text += (
        f"- `{run['run_id']}` — "
        f"learning rate {run['learning_rate']}\n"
    )


document_text += """
## Locked training settings

- AdamW
- Batch size 4096
- Maximum 15 epochs
- Early-stopping patience 4
- Early-stopping minimum delta 0.0002
- Weight decay 0.0001
- Train-only class weights
- Train-only scaler fitting
- One final test evaluation per validation-selected checkpoint

## Selection rule

Pipeline B is selected only when its median paired validation
Macro-F1 improvement is at least 0.001 and it wins at least three
of the four matched model-seed comparisons.

Pipeline A is selected under the symmetric negative condition.

Otherwise, the pipelines are treated as practically equivalent
and Pipeline A is retained. Test-set results are not used for this
selection.
"""


OUTPUT_DOCUMENT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DOCUMENT.write_text(
    document_text,
    encoding="utf-8",
)


manifest_files = [
    OUTPUT_PROTOCOL,
    OUTPUT_DOCUMENT,
    SOURCE_PROTOCOL,
    MODEL_REGISTRY,
    PHASE_1D_SUMMARY,
    SCRIPT_PATH,
]

manifest = {
    "protocol_version":
        "numeric_order_training_pilot_v2_1",

    "created_at":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "files": [],
}


for path in manifest_files:
    manifest["files"].append(
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
    )


OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 78)
print("FAZ 1E PİLOT PROTOKOLÜ")
print("=" * 78)
print("Modeller       : tinyml_mlp, compact_dnn")
print("Seed'ler       : 42, 123")
print("Sayısal hatlar : pipeline_a, pipeline_b")
print("Toplam koşul   : 8")
print("Batch size     : 4096")
print("Max epoch      : 15")
print("Patience       : 4")
print("Min delta      : 0.0002")
print("Weight decay   : 0.0001")
print("TinyML LR      : 0.003")
print("Compact LR     : 0.001")
print()
print("RUN MATRIX")

for run in run_matrix:
    print(
        f"- {run['run_id']} | "
        f"lr={run['learning_rate']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Protokol : {OUTPUT_PROTOCOL}")
print(f"Manifest : {OUTPUT_MANIFEST}")
print(f"Belge    : {OUTPUT_DOCUMENT}")
print()
print("FAZ 1E PİLOT PROTOKOLÜ KİLİTLENDİ")
