"""
N-BaIoT family_3 öğrenme oranı taraması.

Arama uzayı:
- Modeller:
    tinyml_mlp
    compact_dnn
    tiny_1d_cnn
- Öğrenme oranları:
    3e-4
    1e-3
    3e-3
- Seed:
    42
- En fazla epoch:
    10
- Early stopping patience:
    3

Bilimsel kurallar:
- Tam train ve validation splitleri kullanılır.
- Test splitine erişilmez.
- Bütün mimarilere aynı öğrenme oranı arama uzayı verilir.
- Her mimari için öğrenme oranı validation Macro F1 ile seçilir.
- Eşitlik durumunda düşük validation loss ve daha erken best epoch
  tercih edilir.
- Sonuçlar hiperparametre seçimi içindir; nihai makale sonucu değildir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.training.baseline_trainer import run_training  # noqa: E402


DEFAULT_CONFIG_DIRECTORY = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "learning_rate_sweep"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


MODEL_NAMES = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

DEFAULT_LEARNING_RATES = (
    0.0003,
    0.001,
    0.003,
)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Üç TinyML mimarisi için eşit bütçeli "
            "öğrenme oranı taraması çalıştırır."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hiperparametre tarama tohumu.",
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(
            DEFAULT_LEARNING_RATES
        ),
        help="Taranacak öğrenme oranları.",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--config-directory",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    return parser.parse_args()


def json_default(
    value: object,
) -> object:
    """Path gibi nesneleri JSON uyumlu hâle getirir."""

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    temporary_file.replace(
        output_file
    )


def write_yaml_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """YAML yapılandırmasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        yaml.safe_dump(
            data,
            file_handle,
            sort_keys=False,
            allow_unicode=True,
        )

    temporary_file.replace(
        output_file
    )


def read_json(
    input_file: Path,
) -> dict[str, Any]:
    """JSON dosyasını sözlük olarak yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        document = json.load(
            file_handle
        )

    if not isinstance(
        document,
        dict,
    ):
        raise TypeError(
            f"JSON kökü sözlük değil: {input_file}"
        )

    return document


def format_learning_rate(
    learning_rate: float,
) -> str:
    """Öğrenme oranını dosya adına uygun hâle getirir."""

    return (
        f"{learning_rate:.0e}"
        .replace(
            "+",
            "",
        )
        .replace(
            "-",
            "m",
        )
    )


def build_tuning_config(
    model_name: str,
    learning_rate: float,
    seed: int,
    batch_size: int,
    maximum_epochs: int,
    weight_decay: float,
    patience: int,
    minimum_improvement: float,
    torch_threads: int,
) -> dict[str, Any]:
    """Tek bir model ve öğrenme oranı için yapılandırma oluşturur."""

    learning_rate_tag = (
        format_learning_rate(
            learning_rate
        )
    )

    experiment_name = (
        f"tuning_family3_{model_name}_"
        f"lr{learning_rate_tag}_seed{seed}"
    )

    return {
        "experiment": {
            "name": experiment_name,
            "mode": "tuning",
            "task": "family_3",
            "model": model_name,
            "seed": int(
                seed
            ),
        },
        "data": {
            "batch_size": int(
                batch_size
            ),
            "num_workers": 0,
        },
        "training": {
            "max_epochs": int(
                maximum_epochs
            ),
            "learning_rate": float(
                learning_rate
            ),
            "weight_decay": float(
                weight_decay
            ),
            "gradient_clip_norm": 1.0,
            "patience": int(
                patience
            ),
            "min_delta": float(
                minimum_improvement
            ),
            "torch_threads": int(
                torch_threads
            ),
            "weight_scheme": (
                "inverse_square_root_frequency_mean1"
            ),

            # Tam train ve validation splitleri kullanılır.
            # Batch sınırı tanımlanmamıştır.
        },
        "output": {
            "directory": (
                "results/experiments/"
                f"{experiment_name}"
            ),
            "overwrite": True,
        },
    }


def validate_arguments(
    args: argparse.Namespace,
) -> tuple[float, ...]:
    """Komut satırı parametrelerini doğrular."""

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs sıfırdan büyük olmalıdır."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size sıfırdan büyük olmalıdır."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "weight-decay negatif olamaz."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience sıfırdan büyük olmalıdır."
        )

    if args.min_delta < 0:
        raise ValueError(
            "min-delta negatif olamaz."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads sıfırdan büyük olmalıdır."
        )

    learning_rates = tuple(
        float(learning_rate)
        for learning_rate
        in args.learning_rates
    )

    if not learning_rates:
        raise ValueError(
            "En az bir öğrenme oranı verilmelidir."
        )

    if any(
        learning_rate <= 0
        for learning_rate
        in learning_rates
    ):
        raise ValueError(
            "Bütün öğrenme oranları pozitif olmalıdır."
        )

    if len(
        learning_rates
    ) != len(
        set(
            learning_rates
        )
    ):
        raise ValueError(
            "Öğrenme oranları tekrar etmemelidir."
        )

    return learning_rates


def main() -> None:
    """Öğrenme oranı taramasını çalıştırır."""

    args = parse_arguments()

    learning_rates = (
        validate_arguments(
            args
        )
    )

    config_directory = (
        args.config_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_run_count = (
        len(MODEL_NAMES)
        * len(learning_rates)
    )

    print("=" * 78)
    print("N-BaIoT Family-3 Öğrenme Oranı Taraması")
    print("=" * 78)
    print(
        "Modeller          : "
        + ", ".join(
            MODEL_NAMES
        )
    )
    print(
        "Öğrenme oranları  : "
        + ", ".join(
            f"{learning_rate:g}"
            for learning_rate
            in learning_rates
        )
    )
    print(
        f"Toplam çalışma    : {total_run_count}"
    )
    print(f"Seed              : {args.seed}")
    print(
        f"En fazla epoch    : {args.max_epochs}"
    )
    print(f"Patience          : {args.patience}")
    print(
        f"Batch boyutu      : {args.batch_size:,}"
    )
    print("Tam train         : Evet")
    print("Tam validation    : Evet")
    print("Test              : Kullanılmayacak")
    print("=" * 78)

    run_records: list[
        dict[str, Any]
    ] = []

    run_index = 0

    for model_name in MODEL_NAMES:
        for learning_rate in learning_rates:
            run_index += 1

            configuration = (
                build_tuning_config(
                    model_name=model_name,
                    learning_rate=(
                        learning_rate
                    ),
                    seed=args.seed,
                    batch_size=(
                        args.batch_size
                    ),
                    maximum_epochs=(
                        args.max_epochs
                    ),
                    weight_decay=(
                        args.weight_decay
                    ),
                    patience=(
                        args.patience
                    ),
                    minimum_improvement=(
                        args.min_delta
                    ),
                    torch_threads=(
                        args.torch_threads
                    ),
                )
            )

            learning_rate_tag = (
                format_learning_rate(
                    learning_rate
                )
            )

            config_file = (
                config_directory
                / (
                    f"family3_{model_name}_"
                    f"lr{learning_rate_tag}_"
                    f"seed{args.seed}.yaml"
                )
            )

            write_yaml_atomic(
                data=configuration,
                output_file=config_file,
            )

            print()
            print("=" * 78)
            print(
                f"Çalışma {run_index}/{total_run_count}"
            )
            print(
                f"Model           : {model_name}"
            )
            print(
                f"Öğrenme oranı   : {learning_rate:g}"
            )
            print("=" * 78)

            run_summary = run_training(
                config_file=config_file
            )

            if bool(
                run_summary[
                    "test_split_evaluated"
                ]
            ):
                raise RuntimeError(
                    "Hiperparametre taramasında "
                    "test splitine erişildi."
                )

            best_metrics_file = Path(
                str(
                    run_summary[
                        "best_validation_metrics_file"
                    ]
                )
            )

            best_metrics = read_json(
                best_metrics_file
            )

            validation_macro_f1 = float(
                best_metrics[
                    "macro_f1"
                ]
            )

            validation_loss = float(
                best_metrics[
                    "loss"
                ]
            )

            validation_accuracy = float(
                best_metrics[
                    "accuracy"
                ]
            )

            validation_mcc = float(
                best_metrics[
                    "matthews_correlation_coefficient"
                ]
            )

            if abs(
                validation_macro_f1
                - float(
                    run_summary[
                        "best_validation_macro_f1"
                    ]
                )
            ) > 1e-12:
                raise RuntimeError(
                    "Run summary ile validation metrics "
                    "Macro F1 değerleri uyuşmuyor."
                )

            run_records.append(
                {
                    "model_name": (
                        model_name
                    ),
                    "learning_rate": float(
                        learning_rate
                    ),
                    "seed": int(
                        args.seed
                    ),
                    "best_epoch": int(
                        run_summary[
                            "best_epoch"
                        ]
                    ),
                    "epochs_completed": int(
                        run_summary[
                            "epochs_completed"
                        ]
                    ),
                    "best_validation_macro_f1": (
                        validation_macro_f1
                    ),
                    "best_validation_loss": (
                        validation_loss
                    ),
                    "best_validation_accuracy": (
                        validation_accuracy
                    ),
                    "best_validation_mcc": (
                        validation_mcc
                    ),
                    "parameter_count": int(
                        run_summary[
                            "parameter_count"
                        ]
                    ),
                    "total_training_seconds": float(
                        run_summary[
                            "total_training_seconds"
                        ]
                    ),
                    "checkpoint_size_bytes": int(
                        run_summary[
                            "checkpoint_size_bytes"
                        ]
                    ),
                    "test_split_evaluated": False,
                    "scientific_result_status": str(
                        run_summary[
                            "scientific_result_status"
                        ]
                    ),
                    "config_file": str(
                        config_file
                    ),
                    "checkpoint_file": str(
                        run_summary[
                            "checkpoint_file"
                        ]
                    ),
                    "run_summary_file": str(
                        Path(
                            run_summary[
                                "checkpoint_file"
                            ]
                        ).parent
                        / "run_summary.json"
                    ),
                }
            )

    all_runs_frame = pd.DataFrame(
        run_records
    )

    if len(
        all_runs_frame
    ) != total_run_count:
        raise RuntimeError(
            "Tamamlanan tarama sayısı beklenen değerle uyuşmuyor."
        )

    ranked_frames: list[
        pd.DataFrame
    ] = []

    selected_records: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_NAMES:
        model_frame = (
            all_runs_frame[
                all_runs_frame[
                    "model_name"
                ]
                == model_name
            ]
            .sort_values(
                by=[
                    "best_validation_macro_f1",
                    "best_validation_loss",
                    "best_epoch",
                    "learning_rate",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        model_frame.insert(
            0,
            "learning_rate_rank",
            range(
                1,
                len(
                    model_frame
                )
                + 1,
            ),
        )

        ranked_frames.append(
            model_frame
        )

        selected_record = (
            model_frame.iloc[
                0
            ].to_dict()
        )

        selected_records.append(
            selected_record
        )

    ranked_frame = pd.concat(
        ranked_frames,
        ignore_index=True,
    )

    selected_frame = pd.DataFrame(
        selected_records
    )

    selected_frame = (
        selected_frame.sort_values(
            by=[
                "best_validation_macro_f1",
                "parameter_count",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected_frame.insert(
        0,
        "architecture_validation_rank",
        range(
            1,
            len(
                selected_frame
            )
            + 1,
        ),
    )

    all_results_file = (
        report_directory
        / "nbaiot_family3_learning_rate_sweep_all.csv"
    )

    selected_results_file = (
        report_directory
        / "nbaiot_family3_learning_rate_sweep_selected.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_learning_rate_sweep_summary.json"
    )

    ranked_frame.to_csv(
        all_results_file,
        index=False,
        encoding="utf-8",
    )

    selected_frame.to_csv(
        selected_results_file,
        index=False,
        encoding="utf-8",
    )

    summary_document: dict[
        str,
        Any,
    ] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "task": "family_3",
        "run_type": (
            "equal_budget_learning_rate_sweep"
        ),
        "scientific_result_status": (
            "hyperparameter_tuning_not_final_result"
        ),
        "seed": int(
            args.seed
        ),
        "models": list(
            MODEL_NAMES
        ),
        "learning_rates": list(
            learning_rates
        ),
        "total_run_count": int(
            total_run_count
        ),
        "maximum_epochs": int(
            args.max_epochs
        ),
        "patience": int(
            args.patience
        ),
        "minimum_improvement": float(
            args.min_delta
        ),
        "batch_size": int(
            args.batch_size
        ),
        "weight_decay": float(
            args.weight_decay
        ),
        "weight_scheme": (
            "inverse_square_root_frequency_mean1"
        ),
        "model_selection_split": (
            "validation"
        ),
        "model_selection_metric": (
            "macro_f1"
        ),
        "tie_breaking_policy": [
            "lower validation loss",
            "earlier best epoch",
            "lower learning rate",
        ],
        "test_split_evaluated": False,
        "selected_configuration_by_model": {
            str(
                record[
                    "model_name"
                ]
            ): {
                "learning_rate": float(
                    record[
                        "learning_rate"
                    ]
                ),
                "best_epoch": int(
                    record[
                        "best_epoch"
                    ]
                ),
                "epochs_completed": int(
                    record[
                        "epochs_completed"
                    ]
                ),
                "best_validation_macro_f1": float(
                    record[
                        "best_validation_macro_f1"
                    ]
                ),
                "best_validation_loss": float(
                    record[
                        "best_validation_loss"
                    ]
                ),
                "checkpoint_file": str(
                    record[
                        "checkpoint_file"
                    ]
                ),
                "config_file": str(
                    record[
                        "config_file"
                    ]
                ),
            }
            for record in selected_records
        },
        "all_results_csv": str(
            all_results_file
        ),
        "selected_results_csv": str(
            selected_results_file
        ),
        "interpretation_policy": (
            "Selected learning rates will be frozen before "
            "the multi-seed baseline experiments. These "
            "single-seed tuning scores are not final test results."
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        data=summary_document,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Öğrenme oranı taraması tamamlandı")
    print("=" * 78)

    for row in selected_frame.itertuples(
        index=False
    ):
        print(
            f"{row.architecture_validation_rank}. "
            f"{row.model_name:14s} | "
            f"LR={row.learning_rate:g} | "
            f"best_epoch={row.best_epoch} | "
            f"val_macro_f1="
            f"{row.best_validation_macro_f1:.6f} | "
            f"val_loss="
            f"{row.best_validation_loss:.6f}"
        )

    print()
    print(
        "Tamamlanan çalışma : "
        f"{total_run_count}"
    )
    print(
        "Test değerlendirildi: False"
    )
    print(
        "Sonuç statüsü      : Hiperparametre seçimi, nihai sonuç değil"
    )
    print(
        f"Tüm sonuçlar       : {all_results_file}"
    )
    print(
        f"Seçilen ayarlar    : {selected_results_file}"
    )
    print(
        f"JSON özet          : {summary_file}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()