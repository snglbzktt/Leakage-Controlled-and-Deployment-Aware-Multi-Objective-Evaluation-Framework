"""
N-BaIoT deney görevlerini tanımlar ve uygulanabilirliklerini değerlendirir.

Görevler:
1. multiclass_11:
   11 özgün trafik sınıfı. Yalnızca keşifsel analiz.

2. family_3:
   benign, gafgyt ve mirai.
   Ana TinyML mimari, pruning ve quantization karşılaştırma görevi.

3. binary:
   benign ve attack.
   İkincil saldırı tespit görevi.

Bilimsel kurallar:
- Bütün sınıf sayıları mevcut kanonik splitlerden alınır.
- Ağırlıklar yalnızca train sayılarından hesaplanır.
- Validation ve test sınıf ağırlıklarına katılmaz.
- Her görev için split destek yeterliliği raporlanır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LABEL_MAPPING_FILE = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "nbaiot_label_mapping_seed2026.json"
)

DEFAULT_DISTRIBUTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_final_class_distribution.csv"
)

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT görev eşlemelerini, sınıf ağırlıklarını "
            "ve destek yeterliliklerini oluşturur."
        )
    )

    parser.add_argument(
        "--label-mapping-file",
        type=Path,
        default=DEFAULT_LABEL_MAPPING_FILE,
    )

    parser.add_argument(
        "--distribution-file",
        type=Path,
        default=DEFAULT_DISTRIBUTION_FILE,
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--minimum-train-support",
        type=int,
        default=1_000,
        help=(
            "Ana karşılaştırma için bir hedef sınıfta bulunması "
            "gereken en düşük train örneği."
        ),
    )

    parser.add_argument(
        "--minimum-validation-support",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--minimum-test-support",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ==========================================================
# FILE HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını atomik olarak yazar."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            data,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )

    temporary_file.replace(output_file)


def write_csv_atomic(
    frame: pd.DataFrame,
    output_file: Path,
) -> None:
    """CSV dosyasını atomik olarak yazar."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    frame.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8",
    )

    temporary_file.replace(output_file)


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 değerini hesaplar."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ==========================================================
# INPUT
# ==========================================================

def load_original_classes(
    mapping_file: Path,
    expected_seed: int,
) -> list[str]:
    """Nihai 11 sınıflı etiket sırasını yükler."""

    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Etiket eşleme dosyası bulunamadı: {mapping_file}"
        )

    with mapping_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        mapping = json.load(file_handle)

    actual_seed = int(mapping["seed"])

    if actual_seed != expected_seed:
        raise ValueError(
            "Etiket eşleme seed değeri uyuşmuyor: "
            f"dosya={actual_seed}, beklenen={expected_seed}"
        )

    classes = [
        str(class_label)
        for class_label in mapping["class_names"]
    ]

    if not classes:
        raise ValueError(
            "Etiket eşleme dosyasındaki sınıf listesi boş."
        )

    if len(classes) != len(set(classes)):
        raise ValueError(
            "Etiket eşleme dosyasında tekrar eden sınıf var."
        )

    return classes


def load_distribution(
    distribution_file: Path,
    expected_classes: list[str],
) -> pd.DataFrame:
    """15 numaralı betiğin sınıf dağılımını yükler."""

    if not distribution_file.exists():
        raise FileNotFoundError(
            "Sınıf dağılımı bulunamadı: "
            f"{distribution_file}"
        )

    frame = pd.read_csv(distribution_file)

    required_columns = {
        "class_index",
        "class_label",
        "split",
        "sample_count",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dağılım raporunda eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame[
        [
            "class_index",
            "class_label",
            "split",
            "sample_count",
        ]
    ].copy()

    frame["class_index"] = pd.to_numeric(
        frame["class_index"],
        errors="raise",
    ).astype("int64")

    frame["class_label"] = (
        frame["class_label"]
        .astype(str)
    )

    frame["split"] = (
        frame["split"]
        .astype(str)
    )

    frame["sample_count"] = pd.to_numeric(
        frame["sample_count"],
        errors="raise",
    ).astype("int64")

    invalid_splits = (
        set(frame["split"])
        - set(SPLIT_NAMES)
    )

    if invalid_splits:
        raise ValueError(
            "Geçersiz split adları bulundu: "
            + ", ".join(sorted(invalid_splits))
        )

    actual_classes = (
        frame[
            ["class_index", "class_label"]
        ]
        .drop_duplicates()
        .sort_values("class_index")
        ["class_label"]
        .tolist()
    )

    if actual_classes != expected_classes:
        raise ValueError(
            "Dağılım raporundaki sınıf sırası "
            "etiket eşleme dosyasıyla uyuşmuyor."
        )

    duplicated_rows = frame.duplicated(
        subset=[
            "class_label",
            "split",
        ],
        keep=False,
    )

    if duplicated_rows.any():
        raise ValueError(
            "Dağılım raporunda tekrar eden sınıf-split kaydı var."
        )

    expected_record_count = (
        len(expected_classes)
        * len(SPLIT_NAMES)
    )

    if len(frame) != expected_record_count:
        raise ValueError(
            "Dağılım raporundaki kayıt sayısı uyuşmuyor: "
            f"beklenen={expected_record_count}, "
            f"bulunan={len(frame)}"
        )

    if np.any(
        frame["sample_count"].to_numpy() <= 0
    ):
        raise ValueError(
            "Dağılım raporunda sıfır veya negatif örnek sayısı var."
        )

    return frame


# ==========================================================
# TASK DEFINITIONS
# ==========================================================

def build_task_definitions(
    original_classes: list[str],
) -> dict[str, dict[str, Any]]:
    """Üç deney görevini tanımlar."""

    expected_original_classes = {
        "benign",
        "gafgyt_combo",
        "gafgyt_junk",
        "gafgyt_scan",
        "gafgyt_tcp",
        "gafgyt_udp",
        "mirai_ack",
        "mirai_scan",
        "mirai_syn",
        "mirai_udp",
        "mirai_udpplain",
    }

    if set(original_classes) != expected_original_classes:
        raise ValueError(
            "Beklenmeyen özgün sınıf kümesi bulundu."
        )

    multiclass_mapping = {
        class_label: class_label
        for class_label in original_classes
    }

    family_mapping: dict[str, str] = {}

    binary_mapping: dict[str, str] = {}

    for class_label in original_classes:
        if class_label == "benign":
            family_mapping[class_label] = "benign"
            binary_mapping[class_label] = "benign"

        elif class_label.startswith("gafgyt_"):
            family_mapping[class_label] = "gafgyt"
            binary_mapping[class_label] = "attack"

        elif class_label.startswith("mirai_"):
            family_mapping[class_label] = "mirai"
            binary_mapping[class_label] = "attack"

        else:
            raise ValueError(
                f"Sınıf ailesi belirlenemedi: {class_label}"
            )

    return {
        "multiclass_11": {
            "display_name": "11-class fine-grained classification",
            "role": "exploratory",
            "target_classes": original_classes,
            "source_to_target": multiclass_mapping,
            "recommended_loss": None,
            "primary_metric": "Macro F1",
        },
        "family_3": {
            "display_name": "Benign-Gafgyt-Mirai family classification",
            "role": "primary",
            "target_classes": [
                "benign",
                "gafgyt",
                "mirai",
            ],
            "source_to_target": family_mapping,
            "recommended_loss": (
                "cross_entropy_with_"
                "inverse_square_root_frequency_weights"
            ),
            "primary_metric": "Macro F1",
        },
        "binary": {
            "display_name": "Benign-Attack intrusion detection",
            "role": "secondary",
            "target_classes": [
                "benign",
                "attack",
            ],
            "source_to_target": binary_mapping,
            "recommended_loss": (
                "cross_entropy_with_"
                "inverse_square_root_frequency_weights"
            ),
            "primary_metric": "Macro F1",
        },
    }


# ==========================================================
# WEIGHTS
# ==========================================================

def normalize_mean_one(
    values: np.ndarray,
) -> np.ndarray:
    """Ağırlıkları ortalaması 1 olacak şekilde normalleştirir."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "Ağırlık dizisi geçersiz."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Ağırlıklarda sonlu olmayan değer var."
        )

    if np.any(values <= 0):
        raise ValueError(
            "Bütün ağırlıklar pozitif olmalıdır."
        )

    return values / values.mean()


def calculate_weights(
    train_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Train örnek sayılarına göre ağırlık şemalarını hesaplar."""

    train_counts = np.asarray(
        train_counts,
        dtype=np.float64,
    )

    if np.any(train_counts <= 0):
        raise ValueError(
            "Ağırlık hesabında sıfır destekli sınıf var."
        )

    class_count = len(train_counts)
    train_total = train_counts.sum()

    inverse_frequency = (
        train_total
        / (
            class_count
            * train_counts
        )
    )

    inverse_square_root = np.sqrt(
        inverse_frequency
    )

    return {
        "uniform": normalize_mean_one(
            np.ones_like(train_counts)
        ),
        "inverse_frequency_mean1": normalize_mean_one(
            inverse_frequency
        ),
        "inverse_square_root_frequency_mean1": normalize_mean_one(
            inverse_square_root
        ),
    }


# ==========================================================
# TASK AGGREGATION
# ==========================================================

def aggregate_task_distribution(
    original_distribution: pd.DataFrame,
    task_name: str,
    task_definition: dict[str, Any],
    support_thresholds: dict[str, int],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Bir görev için sınıf sayılarını ve uygulanabilirliği hesaplar."""

    target_classes = [
        str(class_label)
        for class_label in task_definition[
            "target_classes"
        ]
    ]

    source_to_target = {
        str(source): str(target)
        for source, target
        in task_definition[
            "source_to_target"
        ].items()
    }

    working = original_distribution.copy()

    working["target_class"] = (
        working["class_label"]
        .map(source_to_target)
    )

    if working["target_class"].isna().any():
        missing_sources = sorted(
            working.loc[
                working["target_class"].isna(),
                "class_label",
            ].unique().tolist()
        )

        raise ValueError(
            f"{task_name} görevinde eşlenmeyen sınıflar var: "
            + ", ".join(missing_sources)
        )

    grouped = (
        working.groupby(
            [
                "split",
                "target_class",
            ],
            as_index=False,
        )["sample_count"]
        .sum()
    )

    records: list[dict[str, Any]] = []

    counts_by_split: dict[
        str,
        np.ndarray,
    ] = {}

    for split_name in SPLIT_NAMES:
        split_frame = (
            grouped[
                grouped["split"] == split_name
            ]
            .set_index("target_class")
        )

        counts = np.asarray(
            [
                int(
                    split_frame.loc[
                        target_class,
                        "sample_count",
                    ]
                )
                if target_class in split_frame.index
                else 0
                for target_class in target_classes
            ],
            dtype=np.int64,
        )

        counts_by_split[
            split_name
        ] = counts

    train_counts = counts_by_split["train"]

    weight_schemes = calculate_weights(
        train_counts
    )

    all_support_rules_pass = True

    for class_index, target_class in enumerate(
        target_classes
    ):
        class_support_pass = True

        for split_name in SPLIT_NAMES:
            sample_count = int(
                counts_by_split[
                    split_name
                ][class_index]
            )

            minimum_required = int(
                support_thresholds[
                    split_name
                ]
            )

            support_pass = bool(
                sample_count >= minimum_required
            )

            class_support_pass = bool(
                class_support_pass
                and support_pass
            )

            records.append(
                {
                    "task_name": task_name,
                    "task_role": task_definition["role"],
                    "target_class_index": class_index,
                    "target_class": target_class,
                    "split": split_name,
                    "sample_count": sample_count,
                    "minimum_required_support": (
                        minimum_required
                    ),
                    "support_rule_passed": (
                        support_pass
                    ),
                    "uniform_weight": float(
                        weight_schemes[
                            "uniform"
                        ][class_index]
                    ),
                    "inverse_frequency_weight_mean1": float(
                        weight_schemes[
                            "inverse_frequency_mean1"
                        ][class_index]
                    ),
                    "inverse_square_root_weight_mean1": float(
                        weight_schemes[
                            "inverse_square_root_frequency_mean1"
                        ][class_index]
                    ),
                }
            )

        all_support_rules_pass = bool(
            all_support_rules_pass
            and class_support_pass
        )

    maximum_train_count = int(
        train_counts.max()
    )

    minimum_train_count = int(
        train_counts.min()
    )

    imbalance_ratio = float(
        maximum_train_count
        / minimum_train_count
    )

    task_summary = {
        "task_name": task_name,
        "display_name": task_definition[
            "display_name"
        ],
        "role": task_definition[
            "role"
        ],
        "target_classes": target_classes,
        "target_class_count": len(target_classes),
        "source_to_target": source_to_target,
        "target_to_index": {
            target_class: index
            for index, target_class
            in enumerate(target_classes)
        },
        "split_counts": {
            split_name: {
                target_classes[index]: int(
                    counts_by_split[
                        split_name
                    ][index]
                )
                for index in range(
                    len(target_classes)
                )
            }
            for split_name in SPLIT_NAMES
        },
        "split_totals": {
            split_name: int(
                counts_by_split[
                    split_name
                ].sum()
            )
            for split_name in SPLIT_NAMES
        },
        "train_imbalance_ratio_max_to_min": (
            imbalance_ratio
        ),
        "minimum_train_class_support": (
            minimum_train_count
        ),
        "maximum_train_class_support": (
            maximum_train_count
        ),
        "support_rule_passed": (
            all_support_rules_pass
        ),
        "eligible_for_primary_model_comparison": bool(
            all_support_rules_pass
            and task_definition["role"] == "primary"
        ),
        "recommended_loss": task_definition[
            "recommended_loss"
        ],
        "primary_metric": task_definition[
            "primary_metric"
        ],
        "weights_fitted_from": "train_only",
        "weight_schemes": {
            scheme_name: {
                target_classes[index]: float(
                    values[index]
                )
                for index in range(
                    len(target_classes)
                )
            }
            for scheme_name, values
            in weight_schemes.items()
        },
    }

    return (
        pd.DataFrame(records),
        weight_schemes,
        task_summary,
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    label_mapping_file = (
        args.label_mapping_file.resolve()
    )

    distribution_file = (
        args.distribution_file.resolve()
    )

    artifact_dir = (
        args.artifact_dir.resolve()
    )

    report_dir = (
        args.report_dir.resolve()
    )

    thresholds = {
        "train": int(
            args.minimum_train_support
        ),
        "validation": int(
            args.minimum_validation_support
        ),
        "test": int(
            args.minimum_test_support
        ),
    }

    if any(
        threshold <= 0
        for threshold in thresholds.values()
    ):
        raise ValueError(
            "Destek eşikleri sıfırdan büyük olmalıdır."
        )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_definition_file = (
        artifact_dir
        / f"nbaiot_task_definitions_seed{args.seed}.json"
    )

    weight_file = (
        artifact_dir
        / f"nbaiot_task_class_weights_seed{args.seed}.npz"
    )

    feasibility_file = (
        report_dir
        / "nbaiot_task_feasibility.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_task_definition_summary.json"
    )

    output_files = (
        task_definition_file,
        weight_file,
        feasibility_file,
        summary_file,
    )

    existing_files = [
        file_path
        for file_path in output_files
        if file_path.exists()
    ]

    if existing_files and not args.overwrite:
        raise FileExistsError(
            "Çıktı dosyaları zaten mevcut. "
            "--overwrite kullan."
        )

    original_classes = load_original_classes(
        mapping_file=label_mapping_file,
        expected_seed=args.seed,
    )

    original_distribution = load_distribution(
        distribution_file=distribution_file,
        expected_classes=original_classes,
    )

    tasks = build_task_definitions(
        original_classes
    )

    print("=" * 78)
    print("N-BaIoT Deney Görevleri ve Uygulanabilirlik Analizi")
    print("=" * 78)
    print(f"Etiket eşleme   : {label_mapping_file}")
    print(f"Dağılım raporu  : {distribution_file}")
    print(f"Train eşiği     : {thresholds['train']:,}")
    print(f"Validation eşiği: {thresholds['validation']:,}")
    print(f"Test eşiği      : {thresholds['test']:,}")
    print("=" * 78)

    feasibility_frames: list[
        pd.DataFrame
    ] = []

    task_summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    weight_artifact_values: dict[
        str,
        np.ndarray,
    ] = {}

    for task_name, task_definition in tasks.items():
        (
            feasibility_frame,
            weight_schemes,
            task_summary,
        ) = aggregate_task_distribution(
            original_distribution=(
                original_distribution
            ),
            task_name=task_name,
            task_definition=task_definition,
            support_thresholds=thresholds,
        )

        feasibility_frames.append(
            feasibility_frame
        )

        task_summaries[
            task_name
        ] = task_summary

        safe_task_name = task_name.replace(
            "-",
            "_",
        )

        weight_artifact_values[
            f"{safe_task_name}_class_names"
        ] = np.asarray(
            task_summary[
                "target_classes"
            ],
            dtype=np.str_,
        )

        weight_artifact_values[
            f"{safe_task_name}_train_counts"
        ] = np.asarray(
            [
                task_summary[
                    "split_counts"
                ]["train"][
                    class_label
                ]
                for class_label in task_summary[
                    "target_classes"
                ]
            ],
            dtype=np.int64,
        )

        for scheme_name, values in (
            weight_schemes.items()
        ):
            weight_artifact_values[
                f"{safe_task_name}_{scheme_name}"
            ] = np.asarray(
                values,
                dtype=np.float32,
            )

    feasibility_frame = pd.concat(
        feasibility_frames,
        ignore_index=True,
    )

    write_csv_atomic(
        frame=feasibility_frame,
        output_file=feasibility_file,
    )

    task_definition_data: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": (
            f"nbaiot_float32_canonical_seed{args.seed}"
        ),
        "seed": int(
            args.seed
        ),
        "original_classes": (
            original_classes
        ),
        "support_thresholds": (
            thresholds
        ),
        "support_threshold_note": (
            "Project-specific screening thresholds for stable "
            "model comparison; they are not universal statistical laws."
        ),
        "tasks": task_summaries,
        "locked_primary_task": "family_3",
        "locked_secondary_task": "binary",
        "exploratory_task": "multiclass_11",
        "locked_primary_loss": (
            "CrossEntropyLoss using train-only "
            "inverse-square-root-frequency weights."
        ),
        "loss_selection_reason": (
            "Inverse-frequency weighting is rejected because "
            "the 11-class task has extreme support imbalance. "
            "Square-root weighting limits gradient amplification."
        ),
        "primary_model_selection_metric": (
            "Validation Macro F1 on family_3"
        ),
        "final_test_policy": (
            "Test is evaluated only after architecture and "
            "compression decisions are finalized on validation."
        ),
    }

    write_json_atomic(
        data=task_definition_data,
        output_file=task_definition_file,
    )

    temporary_weight_file = (
        weight_file.with_suffix(
            weight_file.suffix + ".tmp"
        )
    )

    with temporary_weight_file.open(
        "wb"
    ) as file_handle:
        np.savez_compressed(
            file_handle,
            seed=np.asarray(
                [args.seed],
                dtype=np.int64,
            ),
            **weight_artifact_values,
        )

    temporary_weight_file.replace(
        weight_file
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": int(
            args.seed
        ),
        "locked_primary_task": "family_3",
        "locked_secondary_task": "binary",
        "exploratory_task": "multiclass_11",
        "locked_primary_loss": (
            "inverse_square_root_frequency_weighted_cross_entropy"
        ),
        "task_support_results": {
            task_name: {
                "role": task_summary["role"],
                "target_class_count": (
                    task_summary[
                        "target_class_count"
                    ]
                ),
                "minimum_train_class_support": (
                    task_summary[
                        "minimum_train_class_support"
                    ]
                ),
                "train_imbalance_ratio_max_to_min": (
                    task_summary[
                        "train_imbalance_ratio_max_to_min"
                    ]
                ),
                "support_rule_passed": (
                    task_summary[
                        "support_rule_passed"
                    ]
                ),
                "eligible_for_primary_model_comparison": (
                    task_summary[
                        "eligible_for_primary_model_comparison"
                    ]
                ),
            }
            for task_name, task_summary
            in task_summaries.items()
        },
        "task_definition_file": str(
            task_definition_file
        ),
        "task_weight_file": str(
            weight_file
        ),
        "feasibility_report": str(
            feasibility_file
        ),
        "task_definition_sha256": (
            calculate_sha256(
                task_definition_file
            )
        ),
        "task_weight_sha256": (
            calculate_sha256(
                weight_file
            )
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Deney görevleri başarıyla kilitlendi")
    print("=" * 78)

    for task_name in (
        "multiclass_11",
        "family_3",
        "binary",
    ):
        task_summary = task_summaries[
            task_name
        ]

        print(
            f"{task_name:14s} | "
            f"rol={task_summary['role']:11s} | "
            f"sınıf={task_summary['target_class_count']:2d} | "
            f"min_train="
            f"{task_summary['minimum_train_class_support']:,} | "
            f"oran="
            f"{task_summary['train_imbalance_ratio_max_to_min']:.4f} | "
            f"destek={task_summary['support_rule_passed']}"
        )

    print()
    print(
        "Ana görev       : family_3"
    )
    print(
        "İkincil görev   : binary"
    )
    print(
        "Keşifsel görev  : multiclass_11"
    )
    print(
        "Ana kayıp       : inverse-square-root weighted cross-entropy"
    )
    print(
        "Model seçimi    : validation Macro F1"
    )
    print()
    print(f"Görev tanımları : {task_definition_file}")
    print(f"Görev ağırlıkları: {weight_file}")
    print(f"Destek raporu   : {feasibility_file}")
    print(f"JSON özet       : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
    