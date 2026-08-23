"""
Fiziksel yapılandırılmış budama motorunu doğrular.

Doğrulama kapsamı:
- Seed 42 FP32 checkpointleri kullanılır.
- Her model P25, P50 ve P75 oranlarında budanır.
- Parametre ve MAC azalmaları doğrulanır.
- İleri geçiş, loss, geri yayılım ve state_dict turu sınanır.
- Fine-tuning yapılmaz.
- Validation ve test splitleri kullanılmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.compression.structured_pruning import (  # noqa: E402
    SUPPORTED_PRUNING_RATIOS,
    calculate_parameter_memory_bytes,
    contains_pruning_masks,
    count_model_parameters,
    physically_prune_model,
    recreate_compact_model,
)
from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)
from src.models.nbaiot_models import create_model  # noqa: E402


DEFAULT_BASELINE_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

DEFAULT_PRUNING_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "pruning"
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
            "Seed-42 FP32 modelleri üzerinde fiziksel "
            "yapılandırılmış budama motorunu doğrular."
        )
    )

    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=DEFAULT_BASELINE_DIRECTORY,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--pruning-directory",
        type=Path,
        default=DEFAULT_PRUNING_DIRECTORY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def json_default(
    value: object,
) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        numeric_value = float(
            value
        )

        if not np.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    document: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        output_file.with_suffix(
            output_file.suffix
            + ".tmp"
        )
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            document,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )

    temporary_file.replace(
        output_file
    )


def write_csv_atomic(
    frame: pd.DataFrame,
    output_file: Path,
) -> None:
    """CSV dosyasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        output_file.with_suffix(
            output_file.suffix
            + ".tmp"
        )
    )

    frame.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8",
    )

    temporary_file.replace(
        output_file
    )


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"SHA-256 girdisi bulunamadı: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file_handle:
        while True:
            block = file_handle.read(
                block_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_checkpoint(
    checkpoint_file: Path,
) -> dict[str, Any]:
    """FP32 checkpointini CPU üzerinde yükler."""

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_file}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint kökü sözlük değil."
        )

    return checkpoint


def calculate_gradient_l2_norm(
    model: nn.Module,
) -> float:
    """Model gradientlerinin birleşik L2 normunu hesaplar."""

    squared_sum = 0.0
    gradient_found = False

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient_found = True

        gradient = (
            parameter.grad.detach()
        )

        if not torch.isfinite(
            gradient
        ).all():
            raise RuntimeError(
                "Gradient içinde NaN veya Inf bulundu."
            )

        squared_sum += float(
            torch.sum(
                gradient.double()
                * gradient.double()
            ).item()
        )

    if not gradient_found:
        raise RuntimeError(
            "Budanmış model gradient üretmedi."
        )

    gradient_norm = float(
        np.sqrt(
            squared_sum
        )
    )

    if not np.isfinite(
        gradient_norm
    ):
        raise RuntimeError(
            "Gradient normu sonlu değil."
        )

    return gradient_norm


def validate_state_roundtrip(
    *,
    compact_model: nn.Module,
    metadata: dict[str, Any],
    features: Tensor,
) -> tuple[
    bool,
    int,
]:
    """Kompakt modelin kaydetme-yükleme turunu doğrular."""

    compact_model.eval()

    with torch.inference_mode():
        expected_logits = (
            compact_model(
                features
            )
        )

    memory_buffer = io.BytesIO()

    torch.save(
        compact_model.state_dict(),
        memory_buffer,
    )

    serialized_size = int(
        memory_buffer.tell()
    )

    if serialized_size <= 0:
        raise RuntimeError(
            "Budanmış model boş state_dict üretti."
        )

    memory_buffer.seek(
        0
    )

    try:
        loaded_state = torch.load(
            memory_buffer,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        loaded_state = torch.load(
            memory_buffer,
            map_location="cpu",
        )

    reconstructed_model = (
        recreate_compact_model(
            metadata
        )
    )

    reconstructed_model.load_state_dict(
        loaded_state,
        strict=True,
    )

    reconstructed_model.eval()

    with torch.inference_mode():
        reconstructed_logits = (
            reconstructed_model(
                features
            )
        )

    matches = bool(
        torch.equal(
            expected_logits,
            reconstructed_logits,
        )
    )

    if not matches:
        raise RuntimeError(
            "State_dict turu aynı logitleri üretmedi."
        )

    return (
        matches,
        serialized_size,
    )


def validate_checkpoint_identity(
    *,
    checkpoint: dict[str, Any],
    model_name: str,
    seed: int,
    class_names: tuple[str, ...],
) -> None:
    """FP32 checkpoint kimliğini doğrular."""

    if str(
        checkpoint.get(
            "model_name"
        )
    ) != model_name:
        raise RuntimeError(
            "Checkpoint model adı uyuşmuyor."
        )

    if str(
        checkpoint.get(
            "task_name"
        )
    ) != "family_3":
        raise RuntimeError(
            "Checkpoint görevi family_3 değil."
        )

    if int(
        checkpoint.get(
            "seed"
        )
    ) != seed:
        raise RuntimeError(
            "Checkpoint seed değeri uyuşmuyor."
        )

    checkpoint_classes = tuple(
        str(class_name)
        for class_name
        in checkpoint.get(
            "class_names",
            [],
        )
    )

    if checkpoint_classes != (
        class_names
    ):
        raise RuntimeError(
            "Checkpoint sınıf sırası veri hattıyla uyuşmuyor."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint içinde model_state_dict yok."
        )


def main() -> None:
    """Ana doğrulama akışı."""

    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size pozitif olmalıdır."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads pozitif olmalıdır."
        )

    torch.set_num_threads(
        args.torch_threads
    )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    baseline_directory = (
        args.baseline_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    pruning_directory = (
        args.pruning_directory.resolve()
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pruning_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_directory
        / "nbaiot_structured_pruning_validation.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_structured_pruning_validation_summary.json"
    )

    engine_manifest_file = (
        pruning_directory
        / "nbaiot_structured_pruning_engine_v1.json"
    )

    output_files = (
        report_file,
        summary_file,
        engine_manifest_file,
    )

    if (
        any(
            output_file.exists()
            for output_file in output_files
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "Budama doğrulama çıktıları mevcut. "
            "--overwrite kullan."
        )

    assets = load_pipeline_assets(
        task_name="family_3",
        weight_scheme=(
            "inverse_square_root_frequency_mean1"
        ),
        expected_seed=2026,
    )

    if assets.feature_count != 115:
        raise RuntimeError(
            "Beklenen özellik sayısı 115 değil."
        )

    if assets.class_count != 3:
        raise RuntimeError(
            "family_3 sınıf sayısı 3 değil."
        )

    if tuple(
        assets.target_classes
    ) != (
        "benign",
        "gafgyt",
        "mirai",
    ):
        raise RuntimeError(
            "family_3 sınıf sırası uyuşmuyor."
        )

    _, train_loader = (
        create_nbaiot_dataloader(
            assets=assets,
            split_name="train",
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            epoch=0,
            num_workers=0,
            pin_memory=False,
        )
    )

    try:
        features, targets = next(
            iter(
                train_loader
            )
        )

    except StopIteration as error:
        raise RuntimeError(
            "Train batchi alınamadı."
        ) from error

    if features.dtype != (
        torch.float32
    ):
        raise TypeError(
            "Train özellikleri float32 değil."
        )

    if targets.dtype != (
        torch.int64
    ):
        raise TypeError(
            "Train hedefleri int64 değil."
        )

    if not torch.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Train batchinde NaN veya Inf var."
        )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    validation_records: list[
        dict[str, Any]
    ] = []

    engine_metadata: dict[
        str,
        dict[str, Any]
    ] = {}

    checkpoint_artifacts: dict[
        str,
        dict[str, Any]
    ] = {}

    print("=" * 78)
    print("N-BaIoT Fiziksel Yapılandırılmış Budama Doğrulaması")
    print("=" * 78)
    print(
        "Modeller        : "
        + ", ".join(
            MODEL_NAMES
        )
    )
    print(
        "Budama düzeyleri: P25, P50, P75"
    )
    print(
        f"Kaynak seed     : {args.seed}"
    )
    print(
        f"Train batchi    : {len(targets):,}"
    )
    print(
        "Fine-tuning     : Yapılmayacak"
    )
    print(
        "Validation      : Kullanılmayacak"
    )
    print(
        "Test            : Kullanılmayacak"
    )
    print("=" * 78)

    for model_name in MODEL_NAMES:
        checkpoint_file = (
            baseline_directory
            / model_name
            / f"seed{args.seed}"
            / "best_checkpoint.pt"
        )

        checkpoint = load_checkpoint(
            checkpoint_file
        )

        validate_checkpoint_identity(
            checkpoint=checkpoint,
            model_name=model_name,
            seed=args.seed,
            class_names=(
                assets.target_classes
            ),
        )

        checkpoint_artifacts[
            model_name
        ] = {
            "path": str(
                checkpoint_file
            ),
            "sha256": calculate_sha256(
                checkpoint_file
            ),
        }

        original_model = create_model(
            model_name=model_name,
            input_features=(
                assets.feature_count
            ),
            num_classes=(
                assets.class_count
            ),
        )

        original_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        original_model.cpu()
        original_model.eval()

        original_parameter_count = (
            count_model_parameters(
                original_model
            )
        )

        previous_parameter_count = (
            original_parameter_count
        )

        previous_mac_count: (
            int | None
        ) = None

        engine_metadata[
            model_name
        ] = {}

        print()
        print(
            f"{model_name}:"
        )

        for pruning_ratio in (
            SUPPORTED_PRUNING_RATIOS
        ):
            (
                compact_model,
                metadata_object,
            ) = physically_prune_model(
                model_name=model_name,
                original_model=(
                    original_model
                ),
                pruning_ratio=(
                    pruning_ratio
                ),
            )

            metadata = (
                metadata_object.to_dict()
            )

            compact_parameter_count = (
                count_model_parameters(
                    compact_model
                )
            )

            compact_mac_count = int(
                metadata_object.compact_weight_layer_macs
            )

            if (
                metadata_object.original_parameter_count
                != original_parameter_count
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "özgün parametre sayısı uyuşmuyor."
                )

            if (
                compact_parameter_count
                != metadata_object.compact_parameter_count
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "kompakt parametre sayısı uyuşmuyor."
                )

            if (
                compact_parameter_count
                >= previous_parameter_count
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "parametre sayısı monoton azalmadı."
                )

            if (
                previous_mac_count
                is not None
                and compact_mac_count
                >= previous_mac_count
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "MAC sayısı monoton azalmadı."
                )

            if contains_pruning_masks(
                compact_model
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "maskeli budama anahtarı bulundu."
                )

            compact_model.train()

            compact_model.zero_grad(
                set_to_none=True
            )

            logits = compact_model(
                features
            )

            expected_shape = (
                len(targets),
                assets.class_count,
            )

            if tuple(
                logits.shape
            ) != expected_shape:
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "çıktı boyutu geçersiz."
                )

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "sonlu olmayan logit üretildi."
                )

            loss = criterion(
                logits,
                targets,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "loss sonlu değil."
                )

            loss.backward()

            gradient_norm = (
                calculate_gradient_l2_norm(
                    compact_model
                )
            )

            (
                state_matches,
                serialized_size,
            ) = validate_state_roundtrip(
                compact_model=compact_model,
                metadata=metadata,
                features=features[:64],
            )

            parameter_memory = (
                calculate_parameter_memory_bytes(
                    compact_model
                )
            )

            if parameter_memory != (
                compact_parameter_count
                * 4
            ):
                raise RuntimeError(
                    f"{model_name}/"
                    f"{metadata_object.pruning_name}: "
                    "FP32 parametre belleği uyuşmuyor."
                )

            validation_records.append(
                {
                    "model_name": (
                        model_name
                    ),
                    "source_seed": int(
                        args.seed
                    ),
                    "pruning_name": (
                        metadata_object.pruning_name
                    ),
                    "requested_pruning_ratio": float(
                        pruning_ratio
                    ),
                    "original_parameter_count": int(
                        metadata_object.original_parameter_count
                    ),
                    "compact_parameter_count": int(
                        compact_parameter_count
                    ),
                    "removed_parameter_count": int(
                        metadata_object.removed_parameter_count
                    ),
                    "parameter_reduction_ratio": float(
                        metadata_object.parameter_reduction_ratio
                    ),
                    "original_weight_layer_macs": int(
                        metadata_object.original_weight_layer_macs
                    ),
                    "compact_weight_layer_macs": int(
                        compact_mac_count
                    ),
                    "removed_weight_layer_macs": int(
                        metadata_object.removed_weight_layer_macs
                    ),
                    "mac_reduction_ratio": float(
                        metadata_object.mac_reduction_ratio
                    ),
                    "fp32_parameter_memory_bytes": int(
                        parameter_memory
                    ),
                    "serialized_state_dict_bytes": int(
                        serialized_size
                    ),
                    "loss_before_finetuning": float(
                        loss.detach().item()
                    ),
                    "gradient_l2_norm": float(
                        gradient_norm
                    ),
                    "logits_are_finite": True,
                    "loss_is_finite": True,
                    "gradients_are_finite": True,
                    "state_roundtrip_matches": bool(
                        state_matches
                    ),
                    "contains_pruning_masks": False,
                    "physical_compaction": True,
                    "fine_tuning_performed": False,
                    "validation_split_evaluated": False,
                    "test_split_evaluated": False,
                    "validation_passed": True,
                }
            )

            engine_metadata[
                model_name
            ][
                metadata_object.pruning_name
            ] = metadata

            previous_parameter_count = (
                compact_parameter_count
            )

            previous_mac_count = (
                compact_mac_count
            )

            print(
                f"  {metadata_object.pruning_name} | "
                f"parametre="
                f"{compact_parameter_count:,} | "
                f"azalma="
                f"{metadata_object.parameter_reduction_ratio * 100:.2f}% | "
                f"MAC="
                f"{compact_mac_count:,} | "
                f"MAC azalma="
                f"{metadata_object.mac_reduction_ratio * 100:.2f}%"
            )

    validation_frame = pd.DataFrame(
        validation_records
    )

    expected_combination_count = (
        len(MODEL_NAMES)
        * len(
            SUPPORTED_PRUNING_RATIOS
        )
    )

    if (
        len(validation_frame)
        != expected_combination_count
    ):
        raise RuntimeError(
            "Doğrulanan kombinasyon sayısı 9 değil."
        )

    if validation_frame.duplicated(
        subset=[
            "model_name",
            "pruning_name",
        ]
    ).any():
        raise RuntimeError(
            "Tekrarlanan model-budama kaydı bulundu."
        )

    write_csv_atomic(
        frame=validation_frame,
        output_file=report_file,
    )

    source_file = (
        PROJECT_ROOT
        / "src"
        / "compression"
        / "structured_pruning.py"
    )

    engine_manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "engine_name": (
            "Physical structured bidirectional L1 pruning"
        ),
        "engine_version": "1.0",
        "source_file": str(
            source_file
        ),
        "source_sha256": calculate_sha256(
            source_file
        ),
        "supported_models": list(
            MODEL_NAMES
        ),
        "supported_pruning_ratios": list(
            SUPPORTED_PRUNING_RATIOS
        ),
        "importance_method": (
            "Normalized incoming plus outgoing L1 magnitude"
        ),
        "physical_compaction": True,
        "mask_based_pruning": False,
        "output_class_units_pruned": False,
        "fine_tuning_performed": False,
        "validation_split_evaluated": False,
        "test_split_evaluated": False,
        "source_seed": int(
            args.seed
        ),
        "source_checkpoints": (
            checkpoint_artifacts
        ),
        "metadata": (
            engine_metadata
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        document=engine_manifest,
        output_file=(
            engine_manifest_file
        ),
    )

    all_parameter_counts_reduced = bool(
        (
            validation_frame[
                "compact_parameter_count"
            ]
            < validation_frame[
                "original_parameter_count"
            ]
        ).all()
    )

    all_mac_counts_reduced = bool(
        (
            validation_frame[
                "compact_weight_layer_macs"
            ]
            < validation_frame[
                "original_weight_layer_macs"
            ]
        ).all()
    )

    all_state_roundtrips_match = bool(
        validation_frame[
            "state_roundtrip_matches"
        ].all()
    )

    all_values_finite = bool(
        np.isfinite(
            validation_frame[
                [
                    "loss_before_finetuning",
                    "gradient_l2_norm",
                ]
            ].to_numpy(
                dtype=np.float64
            )
        ).all()
    )

    if not (
        all_parameter_counts_reduced
        and all_mac_counts_reduced
        and all_state_roundtrips_match
        and all_values_finite
    ):
        raise RuntimeError(
            "Budama motoru toplu doğrulaması başarısız."
        )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_seed": int(
            args.seed
        ),
        "validated_combination_count": int(
            len(validation_frame)
        ),
        "all_parameter_counts_reduced": (
            all_parameter_counts_reduced
        ),
        "all_mac_counts_reduced": (
            all_mac_counts_reduced
        ),
        "all_state_roundtrips_match": (
            all_state_roundtrips_match
        ),
        "all_values_finite": (
            all_values_finite
        ),
        "physical_compaction": True,
        "mask_based_pruning": False,
        "fine_tuning_performed": False,
        "validation_split_evaluated": False,
        "test_split_evaluated": False,
        "validation_report": str(
            report_file
        ),
        "engine_manifest": str(
            engine_manifest_file
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        document=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Yapılandırılmış Budama Motoru Doğrulandı")
    print("=" * 78)
    print(
        "Doğrulanan kombinasyon : "
        f"{len(validation_frame)}"
    )
    print(
        "Parametreler azaldı    : "
        f"{all_parameter_counts_reduced}"
    )
    print(
        "MAC değerleri azaldı   : "
        f"{all_mac_counts_reduced}"
    )
    print(
        "State turları eşleşti  : "
        f"{all_state_roundtrips_match}"
    )
    print(
        "Tüm değerler sonlu     : "
        f"{all_values_finite}"
    )
    print(
        "Fiziksel küçültme      : True"
    )
    print(
        "Maskeli budama          : False"
    )
    print(
        "Fine-tuning yapıldı     : False"
    )
    print(
        "Validation kullanıldı   : False"
    )
    print(
        "Test değerlendirildi    : False"
    )
    print(
        "Doğrulama geçti         : True"
    )
    print()
    print(
        f"CSV raporu      : {report_file}"
    )
    print(
        f"Motor manifesti : {engine_manifest_file}"
    )
    print(
        f"JSON özet       : {summary_file}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()