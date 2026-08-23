"""
Yeniden üretilebilir N-BaIoT baseline eğitim motoru.

Bilimsel kurallar:
- Model seçimi yalnızca validation Macro F1 üzerinden yapılır.
- Test splitine eğitim veya model seçimi sırasında erişilmez.
- Scaler yalnızca train verisinden öğrenilmiş artifacttan yüklenir.
- Sınıf ağırlıkları yalnızca train dağılımından hesaplanmış artifacttan alınır.
- Her epoch için eğitim ve validation karışıklık matrisi kaydedilir.
- En iyi model validation Macro F1 değerine göre saklanır.
- Smoke-test sonuçları nihai bilimsel sonuç olarak kullanılmaz.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from src.data.nbaiot_pipeline import (
    create_nbaiot_dataloader,
    load_pipeline_assets,
)
from src.models.nbaiot_models import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def json_default(value: object) -> object:
    """NumPy, Tensor ve Path türlerini JSON uyumlu hâle getirir."""

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

    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını güvenli biçimde yazar."""

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


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_yaml_config(
    config_file: Path,
) -> dict[str, Any]:
    """YAML deney yapılandırmasını yükler."""

    if not config_file.exists():
        raise FileNotFoundError(
            f"Deney yapılandırması bulunamadı: {config_file}"
        )

    with config_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = yaml.safe_load(file_handle)

    if not isinstance(config, dict):
        raise TypeError(
            "YAML kökü sözlük olmalıdır."
        )

    required_sections = {
        "experiment",
        "data",
        "training",
        "output",
    }

    missing_sections = (
        required_sections
        - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Yapılandırmada eksik bölümler var: "
            + ", ".join(sorted(missing_sections))
        )

    return config


def set_reproducible_environment(
    seed: int,
    torch_threads: int,
) -> None:
    """Python, NumPy ve PyTorch tohumlarını sabitler."""

    if torch_threads <= 0:
        raise ValueError(
            "torch_threads sıfırdan büyük olmalıdır."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(
        torch_threads
    )

    torch.use_deterministic_algorithms(
        True
    )


def update_confusion_matrix(
    confusion_matrix: np.ndarray,
    targets: Tensor,
    predictions: Tensor,
    class_count: int,
) -> None:
    """Batch tahminlerini karışıklık matrisine ekler."""

    target_values = (
        targets.detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    prediction_values = (
        predictions.detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    encoded_pairs = (
        target_values * class_count
        + prediction_values
    )

    batch_matrix = np.bincount(
        encoded_pairs,
        minlength=class_count * class_count,
    ).reshape(
        class_count,
        class_count,
    )

    confusion_matrix += batch_matrix


def calculate_metrics(
    confusion_matrix: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Karışıklık matrisinden çok sınıflı metrikleri hesaplar."""

    matrix = np.asarray(
        confusion_matrix,
        dtype=np.int64,
    )

    class_count = len(class_names)

    if matrix.shape != (
        class_count,
        class_count,
    ):
        raise ValueError(
            "Karışıklık matrisi boyutu sınıf sayısıyla uyuşmuyor."
        )

    total = int(
        matrix.sum()
    )

    true_positive = np.diag(
        matrix
    ).astype(np.float64)

    support = matrix.sum(
        axis=1
    ).astype(np.float64)

    predicted_count = matrix.sum(
        axis=0
    ).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(
            true_positive
        ),
        where=predicted_count > 0,
    )

    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(
            true_positive
        ),
        where=support > 0,
    )

    f1_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            true_positive
        ),
        where=(
            precision + recall
        ) > 0,
    )

    false_negative_rate = (
        1.0 - recall
    )

    accuracy = float(
        true_positive.sum() / total
        if total > 0
        else 0.0
    )

    macro_precision = float(
        precision.mean()
    )

    macro_recall = float(
        recall.mean()
    )

    macro_f1 = float(
        f1_score.mean()
    )

    weighted_f1 = float(
        np.sum(
            f1_score * support
        )
        / support.sum()
        if support.sum() > 0
        else 0.0
    )

    trace = float(
        true_positive.sum()
    )

    row_sums = support
    column_sums = predicted_count
    sample_count = float(total)

    numerator = (
        trace * sample_count
        - float(
            np.dot(
                row_sums,
                column_sums,
            )
        )
    )

    denominator_left = (
        sample_count**2
        - float(
            np.dot(
                column_sums,
                column_sums,
            )
        )
    )

    denominator_right = (
        sample_count**2
        - float(
            np.dot(
                row_sums,
                row_sums,
            )
        )
    )

    denominator = np.sqrt(
        max(
            denominator_left
            * denominator_right,
            0.0,
        )
    )

    matthews_correlation_coefficient = float(
        numerator / denominator
        if denominator > 0
        else 0.0
    )

    per_class = {
        class_names[index]: {
            "support": int(
                support[index]
            ),
            "predicted_count": int(
                predicted_count[index]
            ),
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1_score[index]
            ),
            "false_negative_rate": float(
                false_negative_rate[index]
            ),
        }
        for index in range(
            class_count
        )
    }

    return {
        "sample_count": total,
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "matthews_correlation_coefficient": (
            matthews_correlation_coefficient
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def run_epoch(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    class_names: tuple[str, ...],
    optimizer: AdamW | None,
    gradient_clip_norm: float,
    maximum_batches: int | None,
) -> dict[str, Any]:
    """Bir eğitim veya validation epochu çalıştırır."""

    training_mode = (
        optimizer is not None
    )

    if training_mode:
        model.train()
    else:
        model.eval()

    class_count = len(
        class_names
    )

    confusion_matrix = np.zeros(
        (
            class_count,
            class_count,
        ),
        dtype=np.int64,
    )

    total_loss = 0.0
    total_samples = 0
    processed_batches = 0

    start_time = time.perf_counter()

    for features, targets in data_loader:
        if features.dtype != torch.float32:
            raise TypeError(
                "Model girdisi float32 değil."
            )

        if targets.dtype != torch.int64:
            raise TypeError(
                "Model hedefi int64 değil."
            )

        if training_mode:
            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                targets,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Eğitim sırasında sonlu olmayan loss oluştu."
                )

            loss.backward()

            if gradient_clip_norm > 0:
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )

                if not torch.isfinite(
                    gradient_norm
                ):
                    raise RuntimeError(
                        "Sonlu olmayan gradient normu oluştu."
                    )

            optimizer.step()

        else:
            with torch.inference_mode():
                logits = model(
                    features
                )

                loss = criterion(
                    logits,
                    targets,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Validation sırasında sonlu olmayan loss oluştu."
                )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = int(
            targets.numel()
        )

        total_loss += float(
            loss.detach().item()
        ) * batch_size

        total_samples += batch_size
        processed_batches += 1

        update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            targets=targets,
            predictions=predictions,
            class_count=class_count,
        )

        if (
            maximum_batches is not None
            and processed_batches
            >= maximum_batches
        ):
            break

    elapsed_seconds = float(
        time.perf_counter()
        - start_time
    )

    if total_samples <= 0:
        raise RuntimeError(
            "Epoch sırasında örnek işlenmedi."
        )

    metrics = calculate_metrics(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )

    metrics.update(
        {
            "loss": float(
                total_loss
                / total_samples
            ),
            "processed_batches": int(
                processed_batches
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "throughput_samples_per_second": float(
                total_samples
                / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
        }
    )

    return metrics


def prepare_output_directory(
    output_directory: Path,
    overwrite: bool,
) -> None:
    """Deney çıktı klasörünü oluşturur."""

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Deney çıktı klasörü zaten mevcut: "
                f"{output_directory}"
            )

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def run_training(
    config_file: Path,
) -> dict[str, Any]:
    """YAML yapılandırmasından baseline eğitimi çalıştırır."""

    config_file = config_file.resolve()

    config = load_yaml_config(
        config_file
    )

    experiment_config = config[
        "experiment"
    ]

    data_config = config[
        "data"
    ]

    training_config = config[
        "training"
    ]

    output_config = config[
        "output"
    ]

    experiment_name = str(
        experiment_config["name"]
    )

    task_name = str(
        experiment_config["task"]
    )

    model_name = str(
        experiment_config["model"]
    )

    seed = int(
        experiment_config["seed"]
    )

    run_mode = str(
        experiment_config.get(
            "mode",
            "full",
        )
    )

    batch_size = int(
        data_config["batch_size"]
    )

    num_workers = int(
        data_config.get(
            "num_workers",
            0,
        )
    )

    maximum_epochs = int(
        training_config["max_epochs"]
    )

    learning_rate = float(
        training_config["learning_rate"]
    )

    weight_decay = float(
        training_config["weight_decay"]
    )

    gradient_clip_norm = float(
        training_config.get(
            "gradient_clip_norm",
            1.0,
        )
    )

    patience = int(
        training_config.get(
            "patience",
            maximum_epochs,
        )
    )

    minimum_improvement = float(
        training_config.get(
            "min_delta",
            0.0,
        )
    )

    torch_threads = int(
        training_config.get(
            "torch_threads",
            4,
        )
    )

    weight_scheme = str(
        training_config[
            "weight_scheme"
        ]
    )

    maximum_train_batches_raw = (
        training_config.get(
            "max_train_batches"
        )
    )

    maximum_validation_batches_raw = (
        training_config.get(
            "max_validation_batches"
        )
    )

    maximum_train_batches = (
        None
        if maximum_train_batches_raw is None
        else int(
            maximum_train_batches_raw
        )
    )

    maximum_validation_batches = (
        None
        if maximum_validation_batches_raw is None
        else int(
            maximum_validation_batches_raw
        )
    )

    output_directory_raw = Path(
        str(
            output_config["directory"]
        )
    )

    output_directory = (
        output_directory_raw
        if output_directory_raw.is_absolute()
        else PROJECT_ROOT
        / output_directory_raw
    ).resolve()

    overwrite = bool(
        output_config.get(
            "overwrite",
            False,
        )
    )

    if maximum_epochs <= 0:
        raise ValueError(
            "max_epochs sıfırdan büyük olmalıdır."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate sıfırdan büyük olmalıdır."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay negatif olamaz."
        )

    if patience <= 0:
        raise ValueError(
            "patience sıfırdan büyük olmalıdır."
        )

    prepare_output_directory(
        output_directory=output_directory,
        overwrite=overwrite,
    )

    set_reproducible_environment(
        seed=seed,
        torch_threads=torch_threads,
    )

    assets = load_pipeline_assets(
        task_name=task_name,
        weight_scheme=weight_scheme,
        expected_seed=2026,
    )

    train_dataset, train_loader = (
        create_nbaiot_dataloader(
            assets=assets,
            split_name="train",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            epoch=0,
            num_workers=num_workers,
            pin_memory=False,
        )
    )

    _, validation_loader = (
        create_nbaiot_dataloader(
            assets=assets,
            split_name="validation",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            epoch=0,
            num_workers=num_workers,
            pin_memory=False,
        )
    )

    model = create_model(
        model_name=model_name,
        input_features=assets.feature_count,
        num_classes=assets.class_count,
    )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    print("=" * 78)
    print("N-BaIoT Yeniden Üretilebilir Baseline Eğitimi")
    print("=" * 78)
    print(f"Deney adı       : {experiment_name}")
    print(f"Çalışma modu    : {run_mode}")
    print(f"Görev           : {task_name}")
    print(f"Model           : {model_name}")
    print(f"Seed            : {seed}")
    print(f"Özellik sayısı  : {assets.feature_count}")
    print(f"Sınıf sayısı    : {assets.class_count}")
    print(f"Parametre       : {parameter_count:,}")
    print(f"Batch boyutu    : {batch_size:,}")
    print(f"Epoch           : {maximum_epochs}")
    print(f"Öğrenme oranı   : {learning_rate}")
    print(f"Weight decay    : {weight_decay}")
    print(f"Loss ağırlığı   : {weight_scheme}")
    print(f"Çıktı klasörü   : {output_directory}")
    print("Test spliti     : Kullanılmayacak")
    print("=" * 78)

    history_records: list[
        dict[str, Any]
    ] = []

    best_validation_macro_f1 = (
        -np.inf
    )

    best_epoch = 0
    best_model_state: dict[
        str,
        Tensor,
    ] | None = None

    best_optimizer_state: dict[
        str,
        Any,
    ] | None = None

    best_validation_metrics: dict[
        str,
        Any,
    ] | None = None

    epochs_without_improvement = 0

    training_start_time = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch - 1
        )

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            maximum_batches=maximum_train_batches,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=None,
            gradient_clip_norm=0.0,
            maximum_batches=(
                maximum_validation_batches
            ),
        )

        current_validation_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = bool(
            current_validation_macro_f1
            > (
                best_validation_macro_f1
                + minimum_improvement
            )
        )

        if improved:
            best_validation_macro_f1 = (
                current_validation_macro_f1
            )

            best_epoch = epoch

            best_model_state = {
                key: tensor.detach()
                .cpu()
                .clone()
                for key, tensor
                in model.state_dict().items()
            }

            best_optimizer_state = (
                copy.deepcopy(
                    optimizer.state_dict()
                )
            )

            best_validation_metrics = (
                copy.deepcopy(
                    validation_metrics
                )
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        history_records.append(
            {
                "epoch": int(epoch),
                "learning_rate": float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
                "train_loss": float(
                    train_metrics["loss"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "train_macro_f1": float(
                    train_metrics["macro_f1"]
                ),
                "train_weighted_f1": float(
                    train_metrics[
                        "weighted_f1"
                    ]
                ),
                "validation_loss": float(
                    validation_metrics["loss"]
                ),
                "validation_accuracy": float(
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_macro_f1": float(
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_weighted_f1": float(
                    validation_metrics[
                        "weighted_f1"
                    ]
                ),
                "validation_mcc": float(
                    validation_metrics[
                        "matthews_correlation_coefficient"
                    ]
                ),
                "train_sample_count": int(
                    train_metrics[
                        "sample_count"
                    ]
                ),
                "validation_sample_count": int(
                    validation_metrics[
                        "sample_count"
                    ]
                ),
                "train_elapsed_seconds": float(
                    train_metrics[
                        "elapsed_seconds"
                    ]
                ),
                "validation_elapsed_seconds": float(
                    validation_metrics[
                        "elapsed_seconds"
                    ]
                ),
                "improved": bool(
                    improved
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}/{maximum_epochs:02d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.6f} | "
            f"val_loss={validation_metrics['loss']:.6f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.6f} | "
            f"best={best_validation_macro_f1:.6f}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "Erken durdurma etkinleştirildi."
            )
            break

    total_training_seconds = float(
        time.perf_counter()
        - training_start_time
    )

    if (
        best_model_state is None
        or best_optimizer_state is None
        or best_validation_metrics is None
    ):
        raise RuntimeError(
            "En iyi model durumu oluşturulamadı."
        )

    model.load_state_dict(
        best_model_state,
        strict=True,
    )

    history_file = (
        output_directory
        / "training_history.csv"
    )

    best_metrics_file = (
        output_directory
        / "best_validation_metrics.json"
    )

    checkpoint_file = (
        output_directory
        / "best_checkpoint.pt"
    )

    summary_file = (
        output_directory
        / "run_summary.json"
    )

    pd.DataFrame(
        history_records
    ).to_csv(
        history_file,
        index=False,
        encoding="utf-8",
    )

    write_json_atomic(
        data=best_validation_metrics,
        output_file=best_metrics_file,
    )

    checkpoint = {
        "format_version": 1,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_name": (
            experiment_name
        ),
        "run_mode": run_mode,
        "model_name": model_name,
        "task_name": task_name,
        "seed": seed,
        "feature_count": int(
            assets.feature_count
        ),
        "class_names": list(
            assets.target_classes
        ),
        "class_weight_scheme": (
            weight_scheme
        ),
        "class_weights": (
            assets.class_weights.tolist()
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_macro_f1": float(
            best_validation_macro_f1
        ),
        "model_state_dict": (
            best_model_state
        ),
        "optimizer_state_dict": (
            best_optimizer_state
        ),
        "config": config,
    }

    temporary_checkpoint = (
        checkpoint_file.with_suffix(
            checkpoint_file.suffix
            + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_checkpoint,
    )

    temporary_checkpoint.replace(
        checkpoint_file
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_name": (
            experiment_name
        ),
        "run_mode": run_mode,
 "scientific_result_status": (
    "smoke_test_not_for_publication"
    if run_mode == "smoke"
    else (
        "pilot_for_protocol_selection_not_final_result"
        if run_mode == "pilot"
        else (
            "hyperparameter_tuning_not_final_result"
            if run_mode == "tuning"
            else "candidate_experiment"
        )
    )
),
        "test_split_evaluated": False,
        "model_selection_split": (
            "validation"
        ),
        "model_selection_metric": (
            "macro_f1"
        ),
        "model_name": model_name,
        "task_name": task_name,
        "seed": int(seed),
        "parameter_count": (
            parameter_count
        ),
        "feature_count": int(
            assets.feature_count
        ),
        "class_count": int(
            assets.class_count
        ),
        "class_names": list(
            assets.target_classes
        ),
        "class_weight_scheme": (
            weight_scheme
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_macro_f1": float(
            best_validation_macro_f1
        ),
        "epochs_completed": int(
            len(history_records)
        ),
        "total_training_seconds": (
            total_training_seconds
        ),
        "config_file": str(
            config_file
        ),
        "config_sha256": (
            calculate_sha256(
                config_file
            )
        ),
        "history_file": str(
            history_file
        ),
        "best_validation_metrics_file": str(
            best_metrics_file
        ),
        "checkpoint_file": str(
            checkpoint_file
        ),
        "checkpoint_size_bytes": int(
            checkpoint_file.stat().st_size
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Baseline eğitim çalışması tamamlandı")
    print("=" * 78)
    print(f"En iyi epoch          : {best_epoch}")
    print(
        "En iyi validation Macro F1: "
        f"{best_validation_macro_f1:.6f}"
    )
    print(
        "Tamamlanan epoch      : "
        f"{len(history_records)}"
    )
    print(
        "Toplam süre           : "
        f"{total_training_seconds:.2f} saniye"
    )
    print("Test değerlendirildi  : False")
    print(f"Checkpoint            : {checkpoint_file}")
    print(f"Geçmiş                : {history_file}")
    print(f"JSON özet             : {summary_file}")
    print("=" * 78)

    return summary