"""
N-BaIoT TinyML model kayıt sistemini doğrular.

Kontroller:
- Üç model kayıt sisteminden oluşturulabiliyor mu?
- Gerçek family_3 veri batchi ile ileri geçiş yapılabiliyor mu?
- Çıktı boyutu [batch, class_count] biçiminde mi?
- Logitler, loss ve gradientler sonlu mu?
- Eval modunda aynı girdi aynı çıktıyı üretiyor mu?
- Model state_dict kopyası aynı çıktıyı üretiyor mu?
- Modeller binary ve multiclass_11 çıkışlarını da destekliyor mu?
- Parametre sayısı ve ağırlık katmanı MAC tahmini hesaplanabiliyor mu?

Çıktılar:
models/architecture/nbaiot_model_registry_v1.json
results/reports/nbaiot_model_registry_validation.csv
results/reports/nbaiot_model_registry_validation_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)

from src.models.nbaiot_models import (  # noqa: E402
    create_model,
    get_model_spec,
    get_registry_manifest,
    list_registered_models,
)


DEFAULT_ARCHITECTURE_DIR = (
    PROJECT_ROOT
    / "models"
    / "architecture"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

MODEL_SOURCE_FILE = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "TinyML model kayıt sistemini gerçek N-BaIoT "
            "veri batchiyle doğrular."
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
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
        "--architecture-dir",
        type=Path,
        default=DEFAULT_ARCHITECTURE_DIR,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def json_default(value: object) -> object:
    """NumPy ve Path türlerini JSON uyumlu hâle getirir."""

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

    temporary_file.replace(
        output_file
    )


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

    temporary_file.replace(
        output_file
    )


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""

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


def set_reproducible_seed(
    seed: int,
) -> None:
    """CPU model oluşturma ve doğrulama tohumlarını ayarlar."""

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int, int]:
    """Toplam, eğitilebilir ve sıfır olmayan parametreleri sayar."""

    total_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    nonzero_parameters = int(
        sum(
            torch.count_nonzero(
                parameter.detach()
            ).item()
            for parameter in model.parameters()
        )
    )

    return (
        total_parameters,
        trainable_parameters,
        nonzero_parameters,
    )


def calculate_state_memory_bytes(
    model: nn.Module,
) -> int:
    """Mevcut state_dict tensörlerinin gerçek byte büyüklüğünü hesaplar."""

    return int(
        sum(
            tensor.numel()
            * tensor.element_size()
            for tensor in model.state_dict().values()
        )
    )


def estimate_weight_layer_macs(
    model: nn.Module,
    example_input: Tensor,
) -> int:
    """
    Linear ve Conv1d ağırlık katmanlarının örnek başına MAC sayısını hesaplar.

    Aktivasyon, pooling, karşılaştırma ve veri taşıma maliyetleri dahil değildir.
    """

    total_macs = 0

    hook_handles: list[
        torch.utils.hooks.RemovableHandle
    ] = []

    def linear_hook(
        module: nn.Linear,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_macs

        total_macs += int(
            module.in_features
            * module.out_features
        )

    def conv1d_hook(
        module: nn.Conv1d,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_macs

        output_length = int(
            output.shape[-1]
        )

        kernel_length = int(
            module.kernel_size[0]
        )

        input_channels_per_group = int(
            module.in_channels
            // module.groups
        )

        total_macs += int(
            module.out_channels
            * output_length
            * input_channels_per_group
            * kernel_length
        )

    for module in model.modules():
        if isinstance(
            module,
            nn.Linear,
        ):
            hook_handles.append(
                module.register_forward_hook(
                    linear_hook
                )
            )

        elif isinstance(
            module,
            nn.Conv1d,
        ):
            hook_handles.append(
                module.register_forward_hook(
                    conv1d_hook
                )
            )

    was_training = model.training

    model.eval()

    try:
        with torch.inference_mode():
            model(
                example_input[:1]
            )

    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

        model.train(
            was_training
        )

    return int(
        total_macs
    )


def calculate_gradient_l2_norm(
    model: nn.Module,
) -> float:
    """Model gradientlerinin birleşik L2 normunu hesaplar."""

    squared_norm = 0.0
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
                "Model gradientinde NaN veya Inf bulundu."
            )

        squared_norm += float(
            torch.sum(
                gradient.double()
                * gradient.double()
            ).item()
        )

    if not gradient_found:
        raise RuntimeError(
            "Model geri yayılımı gradient üretmedi."
        )

    return float(
        math.sqrt(
            squared_norm
        )
    )


def validate_model(
    model_name: str,
    features: Tensor,
    targets: Tensor,
    feature_count: int,
    class_count: int,
    class_weights: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Tek bir kayıtlı modeli doğrular."""

    set_reproducible_seed(
        seed
    )

    model = create_model(
        model_name=model_name,
        input_features=feature_count,
        num_classes=class_count,
    )

    model_spec = get_model_spec(
        model_name
    )

    (
        total_parameters,
        trainable_parameters,
        nonzero_parameters,
    ) = count_parameters(
        model
    )

    state_memory_bytes = (
        calculate_state_memory_bytes(
            model
        )
    )

    theoretical_fp32_bytes = int(
        total_parameters * 4
    )

    theoretical_fp16_bytes = int(
        total_parameters * 2
    )

    theoretical_int8_bytes = int(
        total_parameters
    )

    model.eval()

    with torch.inference_mode():
        logits_first = model(
            features
        )

        logits_second = model(
            features
        )

    expected_shape = (
        features.shape[0],
        class_count,
    )

    if tuple(
        logits_first.shape
    ) != expected_shape:
        raise RuntimeError(
            f"{model_name} çıktı shape uyuşmuyor: "
            f"beklenen={expected_shape}, "
            f"bulunan={tuple(logits_first.shape)}"
        )

    if logits_first.dtype != torch.float32:
        raise TypeError(
            f"{model_name} logit dtype float32 değil: "
            f"{logits_first.dtype}"
        )

    if not torch.isfinite(
        logits_first
    ).all():
        raise RuntimeError(
            f"{model_name} sonlu olmayan logit üretti."
        )

    eval_deterministic = bool(
        torch.equal(
            logits_first,
            logits_second,
        )
    )

    if not eval_deterministic:
        raise RuntimeError(
            f"{model_name} eval modunda deterministik değil."
        )

    copied_state = {
        key: tensor.detach().clone()
        for key, tensor
        in model.state_dict().items()
    }

    set_reproducible_seed(
        seed + 1
    )

    cloned_model = create_model(
        model_name=model_name,
        input_features=feature_count,
        num_classes=class_count,
    )

    cloned_model.load_state_dict(
        copied_state,
        strict=True,
    )

    cloned_model.eval()

    with torch.inference_mode():
        cloned_logits = cloned_model(
            features
        )

    state_roundtrip_matches = bool(
        torch.equal(
            logits_first,
            cloned_logits,
        )
    )

    if not state_roundtrip_matches:
        raise RuntimeError(
            f"{model_name} state_dict kopyası aynı çıktıyı üretmedi."
        )

    weight_tensor = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    training_logits = model(
        features
    )

    training_loss = criterion(
        training_logits,
        targets,
    )

    if not torch.isfinite(
        training_loss
    ):
        raise RuntimeError(
            f"{model_name} sonlu olmayan loss üretti."
        )

    training_loss.backward()

    gradient_l2_norm = (
        calculate_gradient_l2_norm(
            model
        )
    )

    if (
        not np.isfinite(
            gradient_l2_norm
        )
        or gradient_l2_norm <= 0
    ):
        raise RuntimeError(
            f"{model_name} geçersiz gradient normu üretti: "
            f"{gradient_l2_norm}"
        )

    weight_layer_macs = (
        estimate_weight_layer_macs(
            model=model,
            example_input=features,
        )
    )

    compatibility_results: dict[
        str,
        bool,
    ] = {}

    for compatibility_class_count in (
        2,
        3,
        11,
    ):
        set_reproducible_seed(
            seed
            + compatibility_class_count
        )

        compatibility_model = create_model(
            model_name=model_name,
            input_features=feature_count,
            num_classes=(
                compatibility_class_count
            ),
        )

        compatibility_model.eval()

        with torch.inference_mode():
            compatibility_output = (
                compatibility_model(
                    features[:4]
                )
            )

        compatibility_valid = bool(
            tuple(
                compatibility_output.shape
            )
            == (
                min(
                    4,
                    len(features),
                ),
                compatibility_class_count,
            )
            and torch.isfinite(
                compatibility_output
            ).all()
        )

        if not compatibility_valid:
            raise RuntimeError(
                f"{model_name}, "
                f"{compatibility_class_count} sınıflı görevle "
                "uyumlu değil."
            )

        compatibility_results[
            f"class_count_{compatibility_class_count}"
        ] = True

    return {
        "registry_name": model_name,
        "display_name": (
            model_spec.display_name
        ),
        "architecture_family": (
            model_spec.architecture_family
        ),
        "order_sensitive": bool(
            model_spec.order_sensitive
        ),
        "input_feature_count": int(
            feature_count
        ),
        "output_class_count": int(
            class_count
        ),
        "total_parameter_count": int(
            total_parameters
        ),
        "trainable_parameter_count": int(
            trainable_parameters
        ),
        "initial_nonzero_parameter_count": int(
            nonzero_parameters
        ),
        "initial_sparsity_percentage": float(
            (
                total_parameters
                - nonzero_parameters
            )
            / total_parameters
            * 100
            if total_parameters > 0
            else 0.0
        ),
        "state_dict_memory_bytes": int(
            state_memory_bytes
        ),
        "theoretical_fp32_parameter_bytes": int(
            theoretical_fp32_bytes
        ),
        "theoretical_fp16_parameter_bytes": int(
            theoretical_fp16_bytes
        ),
        "theoretical_int8_parameter_bytes": int(
            theoretical_int8_bytes
        ),
        "weight_layer_macs_per_sample": int(
            weight_layer_macs
        ),
        "logit_shape_valid": True,
        "logits_are_finite": True,
        "eval_deterministic": (
            eval_deterministic
        ),
        "state_dict_roundtrip_matches": (
            state_roundtrip_matches
        ),
        "training_loss": float(
            training_loss.detach().item()
        ),
        "training_loss_is_finite": True,
        "gradient_l2_norm": float(
            gradient_l2_norm
        ),
        "gradients_are_finite": True,
        "binary_compatible": (
            compatibility_results[
                "class_count_2"
            ]
        ),
        "family_3_compatible": (
            compatibility_results[
                "class_count_3"
            ]
        ),
        "multiclass_11_compatible": (
            compatibility_results[
                "class_count_11"
            ]
        ),
        "validation_passed": True,
    }


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads sıfırdan büyük olmalıdır."
        )

    torch.set_num_threads(
        args.torch_threads
    )

    architecture_directory = (
        args.architecture_dir.resolve()
    )

    report_directory = (
        args.report_dir.resolve()
    )

    architecture_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_file = (
        architecture_directory
        / "nbaiot_model_registry_v1.json"
    )

    validation_file = (
        report_directory
        / "nbaiot_model_registry_validation.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_model_registry_validation_summary.json"
    )

    output_files = (
        registry_file,
        validation_file,
        summary_file,
    )

    existing_files = [
        file_path
        for file_path in output_files
        if file_path.exists()
    ]

    if existing_files and not args.overwrite:
        raise FileExistsError(
            "Model kayıt çıktıları zaten mevcut. "
            "--overwrite kullan."
        )

    assets = load_pipeline_assets(
        task_name=args.task,
        weight_scheme=args.weight_scheme,
        expected_seed=args.seed,
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
            iter(train_loader)
        )
    except StopIteration as error:
        raise RuntimeError(
            "Model doğrulaması için train batchi alınamadı."
        ) from error

    if features.dtype != torch.float32:
        raise TypeError(
            "Model doğrulama girdisi float32 değil."
        )

    if targets.dtype != torch.int64:
        raise TypeError(
            "Model doğrulama hedefi int64 değil."
        )

    if not torch.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Model doğrulama girdisinde NaN veya Inf var."
        )

    model_names = list_registered_models()

    if model_names != (
        "tinyml_mlp",
        "compact_dnn",
        "tiny_1d_cnn",
    ):
        raise RuntimeError(
            "Model kayıt sırası beklenen sırayla uyuşmuyor."
        )

    print("=" * 78)
    print("N-BaIoT TinyML Model Kayıt Sistemi Doğrulaması")
    print("=" * 78)
    print(f"Görev            : {assets.task_name}")
    print(
        "Hedef sınıflar   : "
        + ", ".join(
            assets.target_classes
        )
    )
    print(f"Özellik sayısı   : {assets.feature_count}")
    print(f"Batch boyutu     : {len(targets):,}")
    print(
        "Kayıtlı modeller : "
        + ", ".join(
            model_names
        )
    )
    print(f"Seed             : {args.seed}")
    print("=" * 78)

    validation_records: list[
        dict[str, Any]
    ] = []

    for model_index, model_name in enumerate(
        model_names
    ):
        print(
            f"{model_name} doğrulanıyor..."
        )

        validation_record = (
            validate_model(
                model_name=model_name,
                features=features,
                targets=targets,
                feature_count=(
                    assets.feature_count
                ),
                class_count=(
                    assets.class_count
                ),
                class_weights=(
                    assets.class_weights
                ),
                seed=(
                    args.seed
                    + model_index * 1000
                ),
            )
        )

        validation_records.append(
            validation_record
        )

    validation_frame = pd.DataFrame(
        validation_records
    )

    write_csv_atomic(
        frame=validation_frame,
        output_file=validation_file,
    )

    model_source_sha256 = (
        calculate_sha256(
            MODEL_SOURCE_FILE
        )
    )

    registry_manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "registry_version": "1.0",
        "registry_status": "locked_before_training",
        "model_source_file": str(
            MODEL_SOURCE_FILE
        ),
        "model_source_sha256": (
            model_source_sha256
        ),
        "dataset": (
            f"nbaiot_float32_canonical_seed{args.seed}"
        ),
        "primary_task": (
            assets.task_name
        ),
        "input_feature_count": int(
            assets.feature_count
        ),
        "primary_output_class_count": int(
            assets.class_count
        ),
        "registered_model_order": list(
            model_names
        ),
        "models": get_registry_manifest(),
        "primary_task_resource_statistics": {
            record[
                "registry_name"
            ]: {
                "total_parameter_count": (
                    record[
                        "total_parameter_count"
                    ]
                ),
                "trainable_parameter_count": (
                    record[
                        "trainable_parameter_count"
                    ]
                ),
                "weight_layer_macs_per_sample": (
                    record[
                        "weight_layer_macs_per_sample"
                    ]
                ),
                "theoretical_fp32_parameter_bytes": (
                    record[
                        "theoretical_fp32_parameter_bytes"
                    ]
                ),
                "theoretical_int8_parameter_bytes": (
                    record[
                        "theoretical_int8_parameter_bytes"
                    ]
                ),
            }
            for record in validation_records
        },
        "mac_definition": (
            "Multiply-accumulate operations from Linear and Conv1d "
            "weight layers only, per individual input sample. "
            "Activations, pooling, comparisons and memory movement "
            "are excluded."
        ),
        "parameter_memory_note": (
            "Theoretical parameter byte counts do not equal final "
            "serialized model size. Quantized bias, metadata and "
            "runtime overhead must be measured separately."
        ),
        "cnn_feature_order_note": (
            "Tiny-1D-CNN is order-sensitive. The scaler artifact "
            "feature order is frozen for all experiments."
        ),
    }

    write_json_atomic(
        data=registry_manifest,
        output_file=registry_file,
    )

    all_models_passed = bool(
        all(
            record[
                "validation_passed"
            ]
            for record in validation_records
        )
    )

    if not all_models_passed:
        raise RuntimeError(
            "Model kayıt sistemindeki modellerden biri doğrulanamadı."
        )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "task_name": assets.task_name,
        "feature_count": int(
            assets.feature_count
        ),
        "class_count": int(
            assets.class_count
        ),
        "registered_model_count": int(
            len(model_names)
        ),
        "registered_models": list(
            model_names
        ),
        "model_validation": (
            validation_records
        ),
        "all_logits_finite": bool(
            all(
                record[
                    "logits_are_finite"
                ]
                for record in validation_records
            )
        ),
        "all_losses_finite": bool(
            all(
                record[
                    "training_loss_is_finite"
                ]
                for record in validation_records
            )
        ),
        "all_gradients_finite": bool(
            all(
                record[
                    "gradients_are_finite"
                ]
                for record in validation_records
            )
        ),
        "all_eval_outputs_deterministic": bool(
            all(
                record[
                    "eval_deterministic"
                ]
                for record in validation_records
            )
        ),
        "all_state_dict_roundtrips_match": bool(
            all(
                record[
                    "state_dict_roundtrip_matches"
                ]
                for record in validation_records
            )
        ),
        "all_tasks_compatible": bool(
            all(
                record[
                    "binary_compatible"
                ]
                and record[
                    "family_3_compatible"
                ]
                and record[
                    "multiclass_11_compatible"
                ]
                for record in validation_records
            )
        ),
        "model_source_sha256": (
            model_source_sha256
        ),
        "registry_file": str(
            registry_file
        ),
        "validation_report": str(
            validation_file
        ),
        "validation_passed": (
            all_models_passed
        ),
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("TinyML model kayıt sistemi doğrulandı")
    print("=" * 78)

    for record in validation_records:
        fp32_kb = (
            record[
                "theoretical_fp32_parameter_bytes"
            ]
            / 1024
        )

        int8_kb = (
            record[
                "theoretical_int8_parameter_bytes"
            ]
            / 1024
        )

        print(
            f"{record['registry_name']:14s} | "
            f"parametre={record['total_parameter_count']:,} | "
            f"MAC={record['weight_layer_macs_per_sample']:,} | "
            f"FP32={fp32_kb:.2f} KB | "
            f"INT8≈{int8_kb:.2f} KB | "
            f"loss={record['training_loss']:.6f}"
        )

    print()
    print(
        "Bütün logitler sonlu       : "
        f"{summary['all_logits_finite']}"
    )
    print(
        "Bütün loss değerleri sonlu : "
        f"{summary['all_losses_finite']}"
    )
    print(
        "Bütün gradientler sonlu     : "
        f"{summary['all_gradients_finite']}"
    )
    print(
        "Eval deterministik          : "
        f"{summary['all_eval_outputs_deterministic']}"
    )
    print(
        "State kopyaları eşleşiyor   : "
        f"{summary['all_state_dict_roundtrips_match']}"
    )
    print(
        "2/3/11 sınıf uyumluluğu     : "
        f"{summary['all_tasks_compatible']}"
    )
    print(
        "Doğrulama geçti             : "
        f"{summary['validation_passed']}"
    )
    print()
    print(f"Model kayıt dosyası: {registry_file}")
    print(f"Doğrulama raporu  : {validation_file}")
    print(f"JSON özet         : {summary_file}")
    print(f"Kaynak SHA-256    : {model_source_sha256}")
    print("=" * 78)


if __name__ == "__main__":
    main()