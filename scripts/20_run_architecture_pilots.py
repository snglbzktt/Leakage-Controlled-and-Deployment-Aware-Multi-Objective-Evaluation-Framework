"""
N-BaIoT family_3 mimari pilot eğitimlerini çalıştırır.

Pilot kapsamı:
- tinyml_mlp
- compact_dnn
- tiny_1d_cnn
- Tek eğitim tohumu: 42
- Tam train ve validation splitleri
- Test splitine erişim yok
- Sabit başlangıç öğrenme oranı: 1e-3
- Validation Macro F1 ile model seçimi

Pilot sonuçları nihai makale sonucu değildir. Amaç:
- Eğitim kararlılığını incelemek
- Yaklaşık epoch ihtiyacını belirlemek
- Mimari bazlı öğrenme davranışını karşılaştırmak
- Sonraki hiperparametre protokolünü oluşturmak
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
    / "pilots"
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


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Üç TinyML mimarisinin tam veri pilot "
            "eğitimlerini sıralı olarak çalıştırır."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Pilot eğitim tohumu.",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=4,
        help="Her pilot için en fazla epoch sayısı.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
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


def json_default(value: object) -> object:
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
    """JSON dosyasını atomik olarak yazar."""

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
    """YAML yapılandırmasını atomik olarak yazar."""

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


def build_pilot_config(
    model_name: str,
    seed: int,
    batch_size: int,
    maximum_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    minimum_improvement: float,
    torch_threads: int,
) -> dict[str, Any]:
    """Tek bir mimari için pilot YAML içeriği üretir."""

    experiment_name = (
        f"pilot_family3_{model_name}_seed{seed}"
    )

    return {
        "experiment": {
            "name": experiment_name,
            "mode": "pilot",
            "task": "family_3",
            "model": model_name,
            "seed": int(seed),
        },
        "data": {
            "batch_size": int(batch_size),
            "num_workers": 0,
        },
        "training": {
            "max_epochs": int(maximum_epochs),
            "learning_rate": float(
                learning_rate
            ),
            "weight_decay": float(
                weight_decay
            ),
            "gradient_clip_norm": 1.0,
            "patience": int(patience),
            "min_delta": float(
                minimum_improvement
            ),
            "torch_threads": int(
                torch_threads
            ),
            "weight_scheme": (
                "inverse_square_root_frequency_mean1"
            ),

            # Tam train ve validation kullanılacaktır.
            # max_train_batches ve max_validation_batches
            # özellikle tanımlanmamıştır.
        },
        "output": {
            "directory": (
                "results/experiments/"
                f"{experiment_name}"
            ),
            "overwrite": True,
        },
    }


def main() -> None:
    """Pilot yapılandırmalarını üretir ve eğitimleri çalıştırır."""

    args = parse_arguments()

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs sıfırdan büyük olmalıdır."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size sıfırdan büyük olmalıdır."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate sıfırdan büyük olmalıdır."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "weight-decay negatif olamaz."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience sıfırdan büyük olmalıdır."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads sıfırdan büyük olmalıdır."
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

    print("=" * 78)
    print("N-BaIoT Family-3 Mimari Pilot Eğitimleri")
    print("=" * 78)
    print(
        "Modeller          : "
        + ", ".join(MODEL_NAMES)
    )
    print(f"Seed              : {args.seed}")
    print(f"Tam train verisi  : Evet")
    print(f"Tam validation    : Evet")
    print(f"Test verisi       : Kullanılmayacak")
    print(f"Batch boyutu      : {args.batch_size:,}")
    print(f"En fazla epoch    : {args.max_epochs}")
    print(
        "Öğrenme oranı     : "
        f"{args.learning_rate}"
    )
    print(
        "Weight decay      : "
        f"{args.weight_decay}"
    )
    print(f"Patience          : {args.patience}")
    print("=" * 78)

    pilot_records: list[
        dict[str, Any]
    ] = []

    config_files: list[
        Path
    ] = []

    for model_name in MODEL_NAMES:
        config = build_pilot_config(
            model_name=model_name,
            seed=args.seed,
            batch_size=args.batch_size,
            maximum_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            minimum_improvement=args.min_delta,
            torch_threads=args.torch_threads,
        )

        config_file = (
            config_directory
            / (
                f"family3_{model_name}_"
                f"pilot_seed{args.seed}.yaml"
            )
        )

        write_yaml_atomic(
            data=config,
            output_file=config_file,
        )

        config_files.append(
            config_file
        )

    for pilot_index, (
        model_name,
        config_file,
    ) in enumerate(
        zip(
            MODEL_NAMES,
            config_files,
            strict=True,
        ),
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"Pilot {pilot_index}/{len(MODEL_NAMES)} "
            f"başlatılıyor: {model_name}"
        )
        print("=" * 78)

        summary = run_training(
            config_file=config_file
        )

        pilot_records.append(
            {
                "model_name": model_name,
                "seed": int(
                    summary["seed"]
                ),
                "parameter_count": int(
                    summary[
                        "parameter_count"
                    ]
                ),
                "best_epoch": int(
                    summary[
                        "best_epoch"
                    ]
                ),
                "epochs_completed": int(
                    summary[
                        "epochs_completed"
                    ]
                ),
                "best_validation_macro_f1": float(
                    summary[
                        "best_validation_macro_f1"
                    ]
                ),
                "total_training_seconds": float(
                    summary[
                        "total_training_seconds"
                    ]
                ),
                "checkpoint_size_bytes": int(
                    summary[
                        "checkpoint_size_bytes"
                    ]
                ),
                "test_split_evaluated": bool(
                    summary[
                        "test_split_evaluated"
                    ]
                ),
                "scientific_result_status": str(
                    summary[
                        "scientific_result_status"
                    ]
                ),
                "config_file": str(
                    config_file
                ),
                "run_summary_file": str(
                    Path(
                        summary[
                            "checkpoint_file"
                        ]
                    ).parent
                    / "run_summary.json"
                ),
                "checkpoint_file": str(
                    summary[
                        "checkpoint_file"
                    ]
                ),
            }
        )

        if summary[
            "test_split_evaluated"
        ]:
            raise RuntimeError(
                f"{model_name} pilotunda test verisine erişildi."
            )

    pilot_frame = pd.DataFrame(
        pilot_records
    )

    pilot_frame = pilot_frame.sort_values(
        by=[
            "best_validation_macro_f1",
            "parameter_count",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    pilot_frame.insert(
        0,
        "validation_rank",
        range(
            1,
            len(pilot_frame) + 1,
        ),
    )

    csv_file = (
        report_directory
        / "nbaiot_family3_architecture_pilot_summary.csv"
    )

    json_file = (
        report_directory
        / "nbaiot_family3_architecture_pilot_summary.json"
    )

    pilot_frame.to_csv(
        csv_file,
        index=False,
        encoding="utf-8",
    )

    summary_document: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "task": "family_3",
        "run_type": (
            "full_train_validation_architecture_pilot"
        ),
        "scientific_result_status": (
            "pilot_for_protocol_selection_not_final_result"
        ),
        "seed": int(
            args.seed
        ),
        "model_selection_split": (
            "validation"
        ),
        "model_selection_metric": (
            "macro_f1"
        ),
        "test_split_evaluated": False,
        "common_hyperparameters": {
            "batch_size": int(
                args.batch_size
            ),
            "max_epochs": int(
                args.max_epochs
            ),
            "learning_rate": float(
                args.learning_rate
            ),
            "weight_decay": float(
                args.weight_decay
            ),
            "patience": int(
                args.patience
            ),
            "min_delta": float(
                args.min_delta
            ),
            "weight_scheme": (
                "inverse_square_root_frequency_mean1"
            ),
        },
        "models": (
            pilot_frame.to_dict(
                orient="records"
            )
        ),
        "highest_validation_macro_f1_model": str(
            pilot_frame.iloc[
                0
            ]["model_name"]
        ),
        "highest_validation_macro_f1": float(
            pilot_frame.iloc[
                0
            ][
                "best_validation_macro_f1"
            ]
        ),
        "interpretation_policy": (
            "This pilot is used only to select the next "
            "learning-rate and epoch protocol. It is not "
            "reported as a final multi-seed test result."
        ),
        "csv_report": str(
            csv_file
        ),
    }

    write_json_atomic(
        data=summary_document,
        output_file=json_file,
    )

    print()
    print("=" * 78)
    print("Üç mimari pilotu tamamlandı")
    print("=" * 78)

    for row in pilot_frame.itertuples(
        index=False
    ):
        print(
            f"{row.validation_rank}. "
            f"{row.model_name:14s} | "
            f"best_epoch={row.best_epoch} | "
            f"val_macro_f1="
            f"{row.best_validation_macro_f1:.6f} | "
            f"süre={row.total_training_seconds:.2f} sn"
        )

    print()
    print(
        "Test değerlendirildi : False"
    )
    print(
        "Pilot statüsü        : Nihai sonuç değil"
    )
    print(f"CSV raporu           : {csv_file}")
    print(f"JSON raporu          : {json_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()