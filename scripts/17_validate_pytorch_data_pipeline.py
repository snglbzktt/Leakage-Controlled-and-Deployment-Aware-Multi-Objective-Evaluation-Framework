"""
N-BaIoT PyTorch veri hattını doğrular.

Kontroller:
- Scaler, görev tanımı ve sınıf ağırlıkları yüklenebiliyor mu?
- Model girdileri [batch, 115] ve float32 mi?
- Etiketler int64 ve geçerli sınıf aralığında mı?
- Standardizasyon sonrasında bütün değerler sonlu mu?
- family_3 sınıf dönüşümü doğru mu?
- Full-scan sırasında split ve sınıf sayıları görev raporuyla uyuşuyor mu?
- Validation sırası tekrarlanabilir mi?
- Train epoch=0 aynı seed ile tekrarlanabilir mi?
- Train epoch değiştiğinde sıralama değişiyor mu?
- Ağırlıklı CrossEntropyLoss sonlu loss ve gradient üretiyor mu?

Çıktılar:
results/reports/nbaiot_pytorch_pipeline_validation_by_split.csv
results/reports/nbaiot_pytorch_pipeline_validation_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.nbaiot_pipeline import (  # noqa: E402
    NBaiotPipelineAssets,
    create_nbaiot_dataloader,
    load_pipeline_assets,
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


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT PyTorch veri yükleme hattını doğrular."
        )
    )

    parser.add_argument(
        "--task",
        type=str,
        default="family_3",
        choices=[
            "family_3",
            "binary",
            "multiclass_11",
        ],
    )

    parser.add_argument(
        "--weight-scheme",
        type=str,
        default=(
            "inverse_square_root_frequency_mean1"
        ),
        choices=[
            "uniform",
            "inverse_frequency_mean1",
            "inverse_square_root_frequency_mean1",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--sample-batches",
        type=int,
        default=8,
        help=(
            "--full-scan kullanılmadığında her splitten "
            "okunacak batch sayısı."
        ),
    )

    parser.add_argument(
        "--full-scan",
        action="store_true",
        help=(
            "Üç splitin tamamını veri hattından geçirerek "
            "sınıf sayılarını birebir doğrular."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    return parser.parse_args()


def json_default(value: object) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if not np.isfinite(
            numeric_value
        ):
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
    """JSON raporunu güvenli biçimde kaydeder."""

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


def validate_batch(
    features: Tensor,
    targets: Tensor,
    assets: NBaiotPipelineAssets,
    split_name: str,
) -> None:
    """Tek bir PyTorch batchinin biçimini doğrular."""

    if not isinstance(
        features,
        Tensor,
    ):
        raise TypeError(
            f"{split_name} özellik çıktısı Tensor değil."
        )

    if not isinstance(
        targets,
        Tensor,
    ):
        raise TypeError(
            f"{split_name} hedef çıktısı Tensor değil."
        )

    if features.ndim != 2:
        raise RuntimeError(
            f"{split_name} özellik tensörü 2 boyutlu değil: "
            f"shape={tuple(features.shape)}"
        )

    if features.shape[1] != (
        assets.feature_count
    ):
        raise RuntimeError(
            f"{split_name} özellik sayısı uyuşmuyor: "
            f"beklenen={assets.feature_count}, "
            f"bulunan={features.shape[1]}"
        )

    if targets.ndim != 1:
        raise RuntimeError(
            f"{split_name} hedef tensörü 1 boyutlu değil."
        )

    if features.shape[0] != (
        targets.shape[0]
    ):
        raise RuntimeError(
            f"{split_name} batch örnek sayıları uyuşmuyor."
        )

    if features.dtype != torch.float32:
        raise TypeError(
            f"{split_name} özellik dtype float32 değil: "
            f"{features.dtype}"
        )

    if targets.dtype != torch.int64:
        raise TypeError(
            f"{split_name} hedef dtype int64 değil: "
            f"{targets.dtype}"
        )

    if features.device.type != "cpu":
        raise RuntimeError(
            f"{split_name} veri hattı başlangıçta CPU "
            "tensörü üretmelidir."
        )

    if not torch.isfinite(
        features
    ).all():
        raise RuntimeError(
            f"{split_name} özelliklerinde NaN veya Inf var."
        )

    if targets.numel() == 0:
        raise RuntimeError(
            f"{split_name} boş batch üretti."
        )

    minimum_target = int(
        targets.min().item()
    )

    maximum_target = int(
        targets.max().item()
    )

    if minimum_target < 0:
        raise RuntimeError(
            f"{split_name} negatif hedef kodu üretti."
        )

    if maximum_target >= (
        assets.class_count
    ):
        raise RuntimeError(
            f"{split_name} hedef kodu sınıf aralığını aşıyor: "
            f"max={maximum_target}, "
            f"class_count={assets.class_count}"
        )


def scan_split(
    assets: NBaiotPipelineAssets,
    split_name: str,
    batch_size: int,
    num_workers: int,
    sample_batches: int,
    full_scan: bool,
    seed: int,
) -> tuple[
    dict[str, Any],
    tuple[Tensor, Tensor] | None,
]:
    """Bir splitin veri hattı çıktısını tarar."""

    dataset, data_loader = (
        create_nbaiot_dataloader(
            assets=assets,
            split_name=split_name,
            batch_size=batch_size,
            shuffle=(
                split_name == "train"
            ),
            seed=seed,
            epoch=0,
            num_workers=num_workers,
            pin_memory=False,
        )
    )

    expected_split_counts = {
        str(class_name): int(count)
        for class_name, count
        in assets.task_definition[
            "split_counts"
        ][split_name].items()
    }

    expected_total_count = int(
        sum(
            expected_split_counts.values()
        )
    )

    if len(dataset) != expected_total_count:
        raise RuntimeError(
            f"{split_name} Dataset uzunluğu görev raporuyla "
            "uyuşmuyor: dataset={len(dataset):,}, "
            f"beklenen={expected_total_count:,}"
        )

    class_counter: Counter[int] = (
        Counter()
    )

    processed_samples = 0
    processed_batches = 0

    global_minimum = np.inf
    global_maximum = -np.inf

    first_batch: tuple[
        Tensor,
        Tensor,
    ] | None = None

    start_time = time.perf_counter()

    progress_total = (
        expected_total_count
        if full_scan
        else min(
            expected_total_count,
            sample_batches
            * batch_size,
        )
    )

    progress = tqdm(
        total=progress_total,
        desc=f"{split_name} PyTorch hattı",
        unit="örnek",
    )

    for features, targets in data_loader:
        validate_batch(
            features=features,
            targets=targets,
            assets=assets,
            split_name=split_name,
        )

        if first_batch is None:
            first_batch = (
                features.clone(),
                targets.clone(),
            )

        batch_sample_count = int(
            targets.numel()
        )

        processed_samples += (
            batch_sample_count
        )

        processed_batches += 1

        minimum_value = float(
            features.min().item()
        )

        maximum_value = float(
            features.max().item()
        )

        global_minimum = min(
            global_minimum,
            minimum_value,
        )

        global_maximum = max(
            global_maximum,
            maximum_value,
        )

        unique_targets, target_counts = (
            torch.unique(
                targets,
                return_counts=True,
            )
        )

        for target_code, count in zip(
            unique_targets.tolist(),
            target_counts.tolist(),
            strict=True,
        ):
            class_counter[
                int(target_code)
            ] += int(count)

        progress.update(
            batch_sample_count
        )

        if (
            not full_scan
            and processed_batches
            >= sample_batches
        ):
            break

    progress.close()

    elapsed_seconds = float(
        time.perf_counter()
        - start_time
    )

    if first_batch is None:
        raise RuntimeError(
            f"{split_name} veri hattı hiç batch üretmedi."
        )

    if full_scan:
        if processed_samples != expected_total_count:
            raise RuntimeError(
                f"{split_name} full-scan örnek sayısı uyuşmuyor: "
                f"beklenen={expected_total_count:,}, "
                f"işlenen={processed_samples:,}"
            )

        for class_index, class_name in enumerate(
            assets.target_classes
        ):
            expected_class_count = int(
                expected_split_counts[
                    class_name
                ]
            )

            actual_class_count = int(
                class_counter[
                    class_index
                ]
            )

            if (
                actual_class_count
                != expected_class_count
            ):
                raise RuntimeError(
                    f"{split_name}/{class_name} sayısı "
                    "uyuşmuyor: "
                    f"beklenen={expected_class_count:,}, "
                    f"bulunan={actual_class_count:,}"
                )

    throughput = float(
        processed_samples
        / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    observed_class_counts = {
        assets.target_classes[
            class_index
        ]: int(
            class_counter[
                class_index
            ]
        )
        for class_index in range(
            assets.class_count
        )
    }

    record = {
        "split": split_name,
        "full_scan": bool(
            full_scan
        ),
        "expected_sample_count": int(
            expected_total_count
        ),
        "processed_sample_count": int(
            processed_samples
        ),
        "processed_batch_count": int(
            processed_batches
        ),
        "batch_size": int(
            batch_size
        ),
        "num_workers": int(
            num_workers
        ),
        "feature_count": int(
            assets.feature_count
        ),
        "class_count": int(
            assets.class_count
        ),
        "feature_dtype": "torch.float32",
        "target_dtype": "torch.int64",
        "all_values_finite": True,
        "minimum_scaled_value": float(
            global_minimum
        ),
        "maximum_scaled_value": float(
            global_maximum
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "throughput_samples_per_second": (
            throughput
        ),
        "observed_class_counts": (
            observed_class_counts
        ),
        "expected_class_counts": (
            expected_split_counts
        ),
        "class_counts_match": bool(
            full_scan
        ),
        "validation_passed": True,
    }

    return (
        record,
        first_batch,
    )


def get_first_batch(
    assets: NBaiotPipelineAssets,
    split_name: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> tuple[Tensor, Tensor]:
    """Deterministik kontrol için ilk batchi döndürür."""

    _, data_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name=split_name,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        num_workers=0,
        pin_memory=False,
    )

    try:
        features, targets = next(
            iter(data_loader)
        )
    except StopIteration as error:
        raise RuntimeError(
            f"{split_name} ilk batchi üretilemedi."
        ) from error

    validate_batch(
        features=features,
        targets=targets,
        assets=assets,
        split_name=split_name,
    )

    return (
        features,
        targets,
    )


def validate_determinism(
    assets: NBaiotPipelineAssets,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Validation ve train karıştırma tekrarlanabilirliğini test eder."""

    validation_features_a, validation_targets_a = (
        get_first_batch(
            assets=assets,
            split_name="validation",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            epoch=0,
        )
    )

    validation_features_b, validation_targets_b = (
        get_first_batch(
            assets=assets,
            split_name="validation",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            epoch=0,
        )
    )

    validation_reproducible = bool(
        torch.equal(
            validation_features_a,
            validation_features_b,
        )
        and torch.equal(
            validation_targets_a,
            validation_targets_b,
        )
    )

    if not validation_reproducible:
        raise RuntimeError(
            "Validation ilk batchi tekrarlanabilir değil."
        )

    train_features_a, train_targets_a = (
        get_first_batch(
            assets=assets,
            split_name="train",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            epoch=0,
        )
    )

    train_features_b, train_targets_b = (
        get_first_batch(
            assets=assets,
            split_name="train",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            epoch=0,
        )
    )

    train_same_epoch_reproducible = bool(
        torch.equal(
            train_features_a,
            train_features_b,
        )
        and torch.equal(
            train_targets_a,
            train_targets_b,
        )
    )

    if not train_same_epoch_reproducible:
        raise RuntimeError(
            "Train aynı epoch ve seed ile tekrarlanabilir değil."
        )

    train_features_epoch_1, train_targets_epoch_1 = (
        get_first_batch(
            assets=assets,
            split_name="train",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            epoch=1,
        )
    )

    train_epoch_changes_order = bool(
        not torch.equal(
            train_features_a,
            train_features_epoch_1,
        )
        or not torch.equal(
            train_targets_a,
            train_targets_epoch_1,
        )
    )

    if not train_epoch_changes_order:
        raise RuntimeError(
            "Train epoch değişmesine rağmen ilk batch değişmedi."
        )

    return {
        "validation_reproducible": (
            validation_reproducible
        ),
        "train_same_epoch_reproducible": (
            train_same_epoch_reproducible
        ),
        "train_epoch_changes_order": (
            train_epoch_changes_order
        ),
        "determinism_validation_passed": True,
    }


def validate_weighted_loss(
    assets: NBaiotPipelineAssets,
    features: Tensor,
    targets: Tensor,
) -> dict[str, Any]:
    """Ağırlıklı CrossEntropyLoss için ileri ve geri geçiş testi yapar."""

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    logits = torch.zeros(
        (
            targets.shape[0],
            assets.class_count,
        ),
        dtype=torch.float32,
        requires_grad=True,
    )

    loss = criterion(
        logits,
        targets,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Ağırlıklı CrossEntropyLoss sonlu değil."
        )

    loss.backward()

    if logits.grad is None:
        raise RuntimeError(
            "Loss geri yayılımı gradient üretmedi."
        )

    if not torch.isfinite(
        logits.grad
    ).all():
        raise RuntimeError(
            "Loss geri yayılımında sonlu olmayan gradient var."
        )

    return {
        "loss_name": (
            "CrossEntropyLoss"
        ),
        "weight_scheme": (
            assets.class_weight_scheme
        ),
        "class_weights": {
            class_name: float(
                assets.class_weights[
                    class_index
                ]
            )
            for class_index, class_name
            in enumerate(
                assets.target_classes
            )
        },
        "class_weight_mean": float(
            assets.class_weights.mean()
        ),
        "smoke_test_loss": float(
            loss.detach().item()
        ),
        "loss_is_finite": True,
        "gradients_are_finite": True,
        "weighted_loss_validation_passed": True,
    }


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num_workers negatif olamaz."
        )

    if args.sample_batches <= 0:
        raise ValueError(
            "sample-batches sıfırdan büyük olmalıdır."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads sıfırdan büyük olmalıdır."
        )

    torch.set_num_threads(
        args.torch_threads
    )

    report_directory = (
        args.report_dir.resolve()
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    assets = load_pipeline_assets(
        task_name=args.task,
        weight_scheme=args.weight_scheme,
        expected_seed=args.seed,
    )

    print("=" * 78)
    print("N-BaIoT PyTorch Veri Hattı Doğrulaması")
    print("=" * 78)
    print(f"Görev             : {assets.task_name}")
    print(
        "Hedef sınıflar    : "
        + ", ".join(
            assets.target_classes
        )
    )
    print(f"Özellik sayısı    : {assets.feature_count}")
    print(f"Batch boyutu      : {args.batch_size:,}")
    print(f"Num workers       : {args.num_workers}")
    print(f"Full scan         : {args.full_scan}")
    print(
        "Ağırlık şeması    : "
        f"{assets.class_weight_scheme}"
    )
    print(f"Seed              : {args.seed}")
    print("=" * 78)

    split_records: list[
        dict[str, Any]
    ] = []

    first_train_batch: tuple[
        Tensor,
        Tensor,
    ] | None = None

    for split_name in SPLIT_NAMES:
        (
            split_record,
            first_batch,
        ) = scan_split(
            assets=assets,
            split_name=split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_batches=args.sample_batches,
            full_scan=args.full_scan,
            seed=args.seed,
        )

        split_records.append(
            split_record
        )

        if split_name == "train":
            first_train_batch = (
                first_batch
            )

    if first_train_batch is None:
        raise RuntimeError(
            "Train batchi doğrulama için alınamadı."
        )

    determinism_result = (
        validate_determinism(
            assets=assets,
            batch_size=min(
                args.batch_size,
                2048,
            ),
            seed=args.seed,
        )
    )

    loss_result = (
        validate_weighted_loss(
            assets=assets,
            features=(
                first_train_batch[0]
            ),
            targets=(
                first_train_batch[1]
            ),
        )
    )

    split_report_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in record.items()
                if key
                not in {
                    "observed_class_counts",
                    "expected_class_counts",
                }
            }
            for record in split_records
        ]
    )

    split_report_file = (
        report_directory
        / "nbaiot_pytorch_pipeline_validation_by_split.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_pytorch_pipeline_validation_summary.json"
    )

    split_report_frame.to_csv(
        split_report_file,
        index=False,
        encoding="utf-8",
    )

    total_processed_samples = int(
        sum(
            record[
                "processed_sample_count"
            ]
            for record in split_records
        )
    )

    all_splits_passed = bool(
        all(
            record[
                "validation_passed"
            ]
            for record in split_records
        )
    )

    full_scan_counts_match = bool(
        (
            not args.full_scan
        )
        or all(
            record[
                "class_counts_match"
            ]
            for record in split_records
        )
    )

    validation_passed = bool(
        all_splits_passed
        and full_scan_counts_match
        and determinism_result[
            "determinism_validation_passed"
        ]
        and loss_result[
            "weighted_loss_validation_passed"
        ]
    )

    if not validation_passed:
        raise RuntimeError(
            "PyTorch veri hattı doğrulaması başarısız."
        )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "task_name": assets.task_name,
        "task_role": assets.task_definition[
            "role"
        ],
        "target_classes": list(
            assets.target_classes
        ),
        "target_to_index": (
            assets.target_to_index
        ),
        "feature_count": int(
            assets.feature_count
        ),
        "feature_dtype": "torch.float32",
        "target_dtype": "torch.int64",
        "scaler_fit_sample_count": int(
            assets.scaler_train_sample_count
        ),
        "scaler_fit_split": "train",
        "class_weight_scheme": (
            assets.class_weight_scheme
        ),
        "batch_size": int(
            args.batch_size
        ),
        "num_workers": int(
            args.num_workers
        ),
        "full_scan": bool(
            args.full_scan
        ),
        "total_processed_samples": int(
            total_processed_samples
        ),
        "split_validation": (
            split_records
        ),
        "determinism_validation": (
            determinism_result
        ),
        "weighted_loss_validation": (
            loss_result
        ),
        "validation_passed": (
            validation_passed
        ),
        "split_report": str(
            split_report_file
        ),
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("PyTorch veri hattı doğrulaması tamamlandı")
    print("=" * 78)

    for record in split_records:
        print(
            f"{record['split']:10s}: "
            f"{record['processed_sample_count']:,} örnek, "
            f"{record['throughput_samples_per_second']:.2f} örnek/sn, "
            f"finite={record['all_values_finite']}"
        )

    print()
    print(
        "Validation tekrarlanabilir : "
        f"{determinism_result['validation_reproducible']}"
    )
    print(
        "Train aynı epoch tekrar    : "
        f"{determinism_result['train_same_epoch_reproducible']}"
    )
    print(
        "Train epoch sırası değişiyor: "
        f"{determinism_result['train_epoch_changes_order']}"
    )
    print(
        "Weighted loss sonlu        : "
        f"{loss_result['loss_is_finite']}"
    )
    print(
        "Gradientler sonlu          : "
        f"{loss_result['gradients_are_finite']}"
    )
    print(
        "Doğrulama geçti            : "
        f"{validation_passed}"
    )
    print()
    print(f"Split raporu : {split_report_file}")
    print(f"JSON özet    : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()