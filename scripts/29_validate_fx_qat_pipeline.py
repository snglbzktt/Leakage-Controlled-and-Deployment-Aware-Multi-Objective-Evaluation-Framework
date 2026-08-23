"""
N-BaIoT family_3 FX Graph Mode QAT motor doğrulaması.

Doğrulanan akış:
1. Seed-42 FP32 checkpointinin yüklenmesi
2. oneDNN QAT qconfig mapping oluşturulması
3. prepare_qat_fx ile fake-quantized model hazırlanması
4. Yalnızca train batchleriyle kısa QAT güncellemesi
5. convert_fx ile gerçek quantized modele dönüşüm
6. Quantized modül/grafik kapsamının doğrulanması
7. FP32, fake-quantized ve dönüştürülmüş INT8 çıktılarının karşılaştırılması

Bilimsel sınırlar:
- Validation kullanılmaz.
- Test kullanılmaz.
- Bu aşamadaki loss ve çıktı uyumu nihai performans sonucu değildir.
- Amaç yalnızca QAT motorunun üç mimaride teknik olarak çalıştığını
  doğrulamaktır.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.fx.proxy import Proxy
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping,
)
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase,
)
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    prepare_qat_fx,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)

import src.models.nbaiot_models as nbaiot_models_module  # noqa: E402
from src.models.nbaiot_models import create_model  # noqa: E402


DEFAULT_BASELINE_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "quantization"
    / "qat_fx_validation_v1"
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

TARGET_CLASSES = (
    "benign",
    "gafgyt",
    "mirai",
)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Seed-42 FP32 modellerinde oneDNN FX-QAT "
            "dönüşüm zincirini doğrular."
        )
    )

    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=DEFAULT_BASELINE_DIRECTORY,
    )

    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="onednn",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--qat-steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
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
    document: dict[str, Any],
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


def save_torch_atomic(
    document: Any,
    output_file: Path,
) -> None:
    """PyTorch artifactını atomik biçimde kaydeder."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    torch.save(
        document,
        temporary_file,
    )

    temporary_file.replace(
        output_file
    )


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 değerini hesaplar."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"SHA-256 girdisi bulunamadı: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
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

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint kökü sözlük değil."
        )

    return checkpoint


def validate_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model_name: str,
    seed: int,
    class_names: tuple[str, ...],
) -> None:
    """Checkpoint kimliğini doğrular."""

    if str(
        checkpoint.get("model_name")
    ) != model_name:
        raise RuntimeError(
            f"{model_name}: checkpoint model adı uyuşmuyor."
        )

    if str(
        checkpoint.get("task_name")
    ) != "family_3":
        raise RuntimeError(
            f"{model_name}: checkpoint görevi family_3 değil."
        )

    if int(
        checkpoint.get("seed")
    ) != seed:
        raise RuntimeError(
            f"{model_name}: checkpoint seed değeri uyuşmuyor."
        )

    checkpoint_classes = tuple(
        str(value)
        for value in checkpoint.get(
            "class_names",
            [],
        )
    )

    if checkpoint_classes != class_names:
        raise RuntimeError(
            f"{model_name}: checkpoint sınıf sırası uyuşmuyor."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{model_name}: model_state_dict bulunamadı."
        )


def resolve_backend_config(
    backend_name: str,
) -> tuple[Any | None, str | None]:
    """
    Kurulu PyTorch sürümündeki backend config üreticisini bulur.

    oneDNN üreticisi bulunamazsa prepare_qat_fx, yalnızca
    qconfig mapping kullanılarak çalıştırılır.
    """

    candidate_locations = (
        (
            "torch.ao.quantization.backend_config",
            f"get_{backend_name}_backend_config",
        ),
        (
            f"torch.ao.quantization.backend_config.{backend_name}",
            f"get_{backend_name}_backend_config",
        ),
    )

    for module_name, function_name in candidate_locations:
        try:
            module = importlib.import_module(
                module_name
            )

            function = getattr(
                module,
                function_name,
            )

            backend_config = function()

            return (
                backend_config,
                f"{module_name}.{function_name}",
            )

        except (
            ImportError,
            AttributeError,
            ModuleNotFoundError,
        ):
            continue

    return (
        None,
        None,
    )


def create_model_from_checkpoint(
    *,
    model_name: str,
    checkpoint: dict[str, Any],
    input_features: int,
    num_classes: int,
) -> nn.Module:
    """FP32 modelini checkpointten yeniden oluşturur."""

    model = create_model(
        model_name=model_name,
        input_features=input_features,
        num_classes=num_classes,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return model.cpu()


def get_train_batch(
    *,
    assets: Any,
    batch_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Train splitinden deterministik bir batch alır."""

    _, train_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="train",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=0,
        pin_memory=False,
    )

    try:
        features, targets = next(
            iter(train_loader)
        )

    except StopIteration as error:
        raise RuntimeError(
            "Train splitinden QAT batchi alınamadı."
        ) from error

    if features.dtype != torch.float32:
        raise TypeError(
            "QAT özellikleri float32 değil."
        )

    if targets.dtype != torch.int64:
        raise TypeError(
            "QAT hedefleri int64 değil."
        )

    if not torch.isfinite(features).all():
        raise RuntimeError(
            "QAT özelliklerinde NaN veya Inf bulundu."
        )

    return (
        features.cpu(),
        targets.cpu(),
    )


def count_fake_quant_modules(
    model: nn.Module,
) -> int:
    """Modeldeki fake-quant modüllerini sayar."""

    return int(
        sum(
            isinstance(
                module,
                FakeQuantizeBase,
            )
            for module in model.modules()
        )
    )


def count_quantized_modules(
    model: nn.Module,
) -> tuple[
    int,
    list[str],
]:
    """Dönüştürülmüş modeldeki quantized modülleri sayar."""

    module_types: list[str] = []

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__
        )

        if (
            ".quantized" in module_path
            or ".intrinsic.quantized" in module_path
        ):
            module_types.append(
                f"{type(module).__module__}.{class_name}"
            )

    return (
        len(module_types),
        sorted(
            set(module_types)
        ),
    )


def count_quantized_graph_nodes(
    model: nn.Module,
) -> tuple[
    int,
    list[str],
]:
    """FX grafiğindeki quantized işlemleri sayar."""

    graph = getattr(
        model,
        "graph",
        None,
    )

    if graph is None:
        return (
            0,
            [],
        )

    quantized_targets: list[str] = []

    for node in graph.nodes:
        target_text = str(
            node.target
        ).lower()

        node_text = str(
            node
        ).lower()

        if (
            "quantized" in target_text
            or "quantize_per_tensor" in target_text
            or "dequantize" in target_text
            or "quantized" in node_text
        ):
            quantized_targets.append(
                str(node.target)
            )

    return (
        len(quantized_targets),
        sorted(
            set(quantized_targets)
        ),
    )


def calculate_gradient_norm(
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

        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                "QAT gradientinde NaN veya Inf bulundu."
            )

        squared_sum += float(
            torch.sum(
                gradient.double()
                * gradient.double()
            ).item()
        )

    if not gradient_found:
        raise RuntimeError(
            "QAT modeli gradient üretmedi."
        )

    gradient_norm = float(
        np.sqrt(
            squared_sum
        )
    )

    if not np.isfinite(gradient_norm):
        raise RuntimeError(
            "QAT gradient normu sonlu değil."
        )

    return gradient_norm


def compare_outputs(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, Any]:
    """İki model çıktısını karşılaştırır."""

    reference = (
        reference_logits.detach()
        .cpu()
        .float()
    )

    candidate = (
        candidate_logits.detach()
        .cpu()
        .float()
    )

    if reference.shape != candidate.shape:
        raise RuntimeError(
            "Çıktı boyutları uyuşmuyor: "
            f"referans={tuple(reference.shape)}, "
            f"aday={tuple(candidate.shape)}"
        )

    if not torch.isfinite(candidate).all():
        raise RuntimeError(
            "Aday model çıktısında NaN veya Inf bulundu."
        )

    absolute_difference = torch.abs(
        candidate - reference
    )

    reference_predictions = torch.argmax(
        reference,
        dim=1,
    )

    candidate_predictions = torch.argmax(
        candidate,
        dim=1,
    )

    agreement = float(
        (
            reference_predictions
            == candidate_predictions
        )
        .float()
        .mean()
        .item()
    )

    return {
        "sample_count": int(
            len(reference)
        ),
        "maximum_absolute_difference": float(
            absolute_difference.max().item()
        ),
        "mean_absolute_difference": float(
            absolute_difference.mean().item()
        ),
        "root_mean_squared_difference": float(
            torch.sqrt(
                torch.mean(
                    torch.square(
                        candidate - reference
                    )
                )
            ).item()
        ),
        "argmax_prediction_agreement": (
            agreement
        ),
    }


def prepare_qat_model(
    *,
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    backend_name: str,
    backend_config: Any | None,
) -> nn.Module:
    """
    Modeli FX Graph Mode QAT eğitimine hazırlar.

    FX sembolik izleme sırasında model girdisi gerçek Tensor yerine
    Proxy nesnesidir. Normal çalışma sırasında kullanılan giriş
    doğrulaması Proxy üzerinde Python kontrol akışı oluşturduğu için
    yalnızca tracing süresince Proxy doğrulaması atlanır.

    Gerçek Tensor girdilerinde özgün doğrulama çalışmaya devam eder.
    Tracing tamamlandıktan sonra özgün doğrulama fonksiyonu mutlaka
    geri yüklenir.
    """

    qconfig_mapping = (
        get_default_qat_qconfig_mapping(
            backend_name
        )
    )

    model.train()

    original_validator = (
        nbaiot_models_module.validate_forward_input
    )

    def fx_safe_validate_forward_input(
        *validator_args: Any,
        **validator_kwargs: Any,
    ) -> Any:
        """
        FX Proxy girdisini doğrulama kontrol akışından geçirmez.

        Gerçek Tensor girdileri özgün doğrulama fonksiyonuna gönderilir.
        """

        if "inputs" in validator_kwargs:
            input_value = validator_kwargs[
                "inputs"
            ]

        elif validator_args:
            input_value = validator_args[0]

        else:
            return original_validator(
                *validator_args,
                **validator_kwargs,
            )

        if isinstance(
            input_value,
            Proxy,
        ):
            return None

        return original_validator(
            *validator_args,
            **validator_kwargs,
        )

    nbaiot_models_module.validate_forward_input = (
        fx_safe_validate_forward_input
    )

    try:
        if backend_config is None:
            prepared_model = prepare_qat_fx(
                model,
                qconfig_mapping,
                example_inputs,
            )

        else:
            prepared_model = prepare_qat_fx(
                model,
                qconfig_mapping,
                example_inputs,
                backend_config=backend_config,
            )

    finally:
        nbaiot_models_module.validate_forward_input = (
            original_validator
        )

    return prepared_model


def convert_qat_model(
    *,
    prepared_model: nn.Module,
    backend_config: Any | None,
) -> nn.Module:
    """Hazırlanmış QAT modelini gerçek quantized modele dönüştürür."""

    prepared_model.eval()

    if backend_config is None:
        converted_model = convert_fx(
            prepared_model
        )

    else:
        converted_model = convert_fx(
            prepared_model,
            backend_config=backend_config,
        )

    converted_model.eval()

    return converted_model


def main() -> None:
    """Üç mimaride FX-QAT zincirini doğrular."""

    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size pozitif olmalıdır."
        )

    if args.qat_steps <= 0:
        raise ValueError(
            "qat-steps pozitif olmalıdır."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate pozitif olmalıdır."
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

    supported_engines = tuple(
        torch.backends.quantized.supported_engines
    )

    if args.backend not in supported_engines:
        raise RuntimeError(
            f"{args.backend} quantized engine desteklenmiyor. "
            f"Desteklenenler={supported_engines}"
        )

    torch.backends.quantized.engine = (
        args.backend
    )

    baseline_directory = (
        args.baseline_directory.resolve()
    )

    artifact_directory = (
        args.artifact_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_directory
        / "nbaiot_fx_qat_pipeline_validation.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_fx_qat_pipeline_validation_summary.json"
    )

    if (
        (
            report_file.exists()
            or summary_file.exists()
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "FX-QAT doğrulama çıktıları mevcut. "
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

    if tuple(
        assets.target_classes
    ) != TARGET_CLASSES:
        raise RuntimeError(
            "family_3 sınıf sırası uyuşmuyor."
        )

    features, targets = get_train_batch(
        assets=assets,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    comparison_features = (
        features[:64]
        .contiguous()
    )

    example_inputs = (
        features[:8].contiguous(),
    )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    (
        backend_config,
        backend_config_source,
    ) = resolve_backend_config(
        args.backend
    )

    print("=" * 78)
    print("N-BaIoT FX Graph Mode QAT Motor Doğrulaması")
    print("=" * 78)
    print(
        f"PyTorch          : {torch.__version__}"
    )
    print(
        f"Quantized engine : {torch.backends.quantized.engine}"
    )
    print(
        "Backend config   : "
        f"{backend_config_source or 'qconfig_mapping_only'}"
    )
    print(
        f"Kaynak seed      : {args.seed}"
    )
    print(
        f"Train batchi     : {len(targets):,}"
    )
    print(
        f"QAT adımı        : {args.qat_steps}"
    )
    print(
        "Validation       : Kullanılmayacak"
    )
    print(
        "Test             : Kullanılmayacak"
    )
    print("=" * 78)

    records: list[
        dict[str, Any]
    ] = []

    model_details: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_name in MODEL_NAMES:
        print()
        print(f"{model_name}:")

        checkpoint_file = (
            baseline_directory
            / model_name
            / f"seed{args.seed}"
            / "best_checkpoint.pt"
        )

        checkpoint = load_checkpoint(
            checkpoint_file
        )

        validate_checkpoint(
            checkpoint=checkpoint,
            model_name=model_name,
            seed=args.seed,
            class_names=(
                assets.target_classes
            ),
        )

        fp32_model = create_model_from_checkpoint(
            model_name=model_name,
            checkpoint=checkpoint,
            input_features=assets.feature_count,
            num_classes=assets.class_count,
        )

        fp32_model.eval()

        with torch.inference_mode():
            fp32_logits = fp32_model(
                comparison_features
            )

        qat_source_model = (
            create_model_from_checkpoint(
                model_name=model_name,
                checkpoint=checkpoint,
                input_features=(
                    assets.feature_count
                ),
                num_classes=(
                    assets.class_count
                ),
            )
        )

        prepared_model = prepare_qat_model(
            model=qat_source_model,
            example_inputs=example_inputs,
            backend_name=args.backend,
            backend_config=backend_config,
        )

        fake_quant_count = (
            count_fake_quant_modules(
                prepared_model
            )
        )

        if fake_quant_count <= 0:
            raise RuntimeError(
                f"{model_name}: prepare_qat_fx "
                "fake-quant modülü oluşturmadı."
            )

        optimizer = torch.optim.AdamW(
            prepared_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        losses: list[float] = []
        gradient_norms: list[float] = []

        prepared_model.train()

        for step in range(
            1,
            args.qat_steps + 1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            logits = prepared_model(
                features
            )

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    f"{model_name}: QAT logitleri sonlu değil."
                )

            loss = criterion(
                logits,
                targets,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"{model_name}: QAT loss değeri sonlu değil."
                )

            loss.backward()

            gradient_norm = (
                calculate_gradient_norm(
                    prepared_model
                )
            )

            torch.nn.utils.clip_grad_norm_(
                prepared_model.parameters(),
                max_norm=(
                    args.gradient_clip_norm
                ),
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach().item()
                )
            )

            gradient_norms.append(
                gradient_norm
            )

            print(
                f"  QAT adımı {step}/{args.qat_steps} | "
                f"loss={losses[-1]:.6f} | "
                f"gradient={gradient_norm:.6f}"
            )

        prepared_model.eval()

        with torch.inference_mode():
            fake_quant_logits = (
                prepared_model(
                    comparison_features
                )
            )

        fake_quant_comparison = (
            compare_outputs(
                fp32_logits,
                fake_quant_logits,
            )
        )

        model_directory = (
            artifact_directory
            / model_name
            / f"seed{args.seed}"
        )

        model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        prepared_state_file = (
            model_directory
            / "prepared_qat_state_dict.pt"
        )

        save_torch_atomic(
            prepared_model.state_dict(),
            prepared_state_file,
        )

        converted_model = (
            convert_qat_model(
                prepared_model=prepared_model,
                backend_config=backend_config,
            )
        )

        with torch.inference_mode():
            quantized_logits = (
                converted_model(
                    comparison_features
                )
            )

        quantized_comparison = (
            compare_outputs(
                fp32_logits,
                quantized_logits,
            )
        )

        (
            quantized_module_count,
            quantized_module_types,
        ) = count_quantized_modules(
            converted_model
        )

        (
            quantized_node_count,
            quantized_node_targets,
        ) = count_quantized_graph_nodes(
            converted_model
        )

        if (
            quantized_module_count <= 0
            and quantized_node_count <= 0
        ):
            raise RuntimeError(
                f"{model_name}: convert_fx sonucunda "
                "quantized modül veya grafik düğümü bulunamadı."
            )

        converted_state_file = (
            model_directory
            / "converted_int8_state_dict.pt"
        )

        save_torch_atomic(
            converted_model.state_dict(),
            converted_state_file,
        )

        graph_file = (
            model_directory
            / "converted_fx_graph.txt"
        )

        graph_file.write_text(
            str(
                converted_model.graph
            ),
            encoding="utf-8",
        )

        metadata_file = (
            model_directory
            / "qat_validation_metadata.json"
        )

        metadata = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_name": model_name,
            "source_seed": int(args.seed),
            "backend": args.backend,
            "backend_config_source": (
                backend_config_source
            ),
            "qat_steps": int(
                args.qat_steps
            ),
            "learning_rate": float(
                args.learning_rate
            ),
            "fake_quant_module_count": int(
                fake_quant_count
            ),
            "quantized_module_count": int(
                quantized_module_count
            ),
            "quantized_module_types": (
                quantized_module_types
            ),
            "quantized_graph_node_count": int(
                quantized_node_count
            ),
            "quantized_graph_targets": (
                quantized_node_targets
            ),
            "fp32_vs_fake_quant": (
                fake_quant_comparison
            ),
            "fp32_vs_converted_int8": (
                quantized_comparison
            ),
            "prepared_state_file": str(
                prepared_state_file
            ),
            "converted_state_file": str(
                converted_state_file
            ),
            "graph_file": str(
                graph_file
            ),
            "validation_split_used": False,
            "test_split_used": False,
            "validation_passed": True,
        }

        write_json_atomic(
            metadata,
            metadata_file,
        )

        records.append(
            {
                "model_name": model_name,
                "source_seed": int(
                    args.seed
                ),
                "backend": args.backend,
                "backend_config_source": (
                    backend_config_source
                    or "qconfig_mapping_only"
                ),
                "qat_steps": int(
                    args.qat_steps
                ),
                "final_loss": float(
                    losses[-1]
                ),
                "maximum_gradient_l2_norm": float(
                    max(gradient_norms)
                ),
                "fake_quant_module_count": int(
                    fake_quant_count
                ),
                "quantized_module_count": int(
                    quantized_module_count
                ),
                "quantized_graph_node_count": int(
                    quantized_node_count
                ),
                "fp32_fake_quant_argmax_agreement": float(
                    fake_quant_comparison[
                        "argmax_prediction_agreement"
                    ]
                ),
                "fp32_int8_argmax_agreement": float(
                    quantized_comparison[
                        "argmax_prediction_agreement"
                    ]
                ),
                "fp32_int8_max_abs_difference": float(
                    quantized_comparison[
                        "maximum_absolute_difference"
                    ]
                ),
                "prepared_state_size_bytes": int(
                    prepared_state_file.stat().st_size
                ),
                "converted_state_size_bytes": int(
                    converted_state_file.stat().st_size
                ),
                "validation_split_used": False,
                "test_split_used": False,
                "validation_passed": True,
            }
        )

        model_details[
            model_name
        ] = metadata

        print(
            "  Fake-quant modülü : "
            f"{fake_quant_count}"
        )
        print(
            "  Quantized modül    : "
            f"{quantized_module_count}"
        )
        print(
            "  Quantized düğüm    : "
            f"{quantized_node_count}"
        )
        print(
            "  FP32–INT8 argmax   : "
            f"{quantized_comparison['argmax_prediction_agreement']:.6f}"
        )
        print(
            "  Dönüşüm            : Başarılı"
        )

    report_frame = pd.DataFrame(
        records
    )

    if len(report_frame) != 3:
        raise RuntimeError(
            "Doğrulanan model sayısı 3 değil."
        )

    if not report_frame[
        "validation_passed"
    ].all():
        raise RuntimeError(
            "QAT doğrulamalarından biri başarısız."
        )

    if (
        report_frame[
            "fake_quant_module_count"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Modellerden birinde fake-quant kapsamı yok."
        )

    quantized_coverage = (
        report_frame[
            "quantized_module_count"
        ]
        + report_frame[
            "quantized_graph_node_count"
        ]
    )

    if (
        quantized_coverage <= 0
    ).any():
        raise RuntimeError(
            "Modellerden birinde gerçek quantized kapsam yok."
        )

    write_csv_atomic(
        report_frame,
        report_file,
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT FX Graph Mode QAT pipeline validation"
        ),
        "analysis_version": "1.0",
        "torch_version": str(
            torch.__version__
        ),
        "backend": args.backend,
        "supported_engines": list(
            supported_engines
        ),
        "backend_config_source": (
            backend_config_source
        ),
        "source_seed": int(
            args.seed
        ),
        "models": list(
            MODEL_NAMES
        ),
        "validated_model_count": int(
            len(report_frame)
        ),
        "all_models_have_fake_quant_modules": bool(
            (
                report_frame[
                    "fake_quant_module_count"
                ]
                > 0
            ).all()
        ),
        "all_models_have_quantized_coverage": bool(
            (
                quantized_coverage
                > 0
            ).all()
        ),
        "all_outputs_finite": True,
        "validation_split_used": False,
        "test_split_used": False,
        "model_details": model_details,
        "report_file": str(
            report_file
        ),
        "artifact_directory": str(
            artifact_directory
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        summary,
        summary_file,
    )

    print()
    print("=" * 78)
    print("FX-QAT Motor Doğrulaması Tamamlandı")
    print("=" * 78)
    print(
        "Doğrulanan model          : "
        f"{len(report_frame)}/3"
    )
    print(
        "Fake-quant kapsamı        : True"
    )
    print(
        "Gerçek quantized kapsam   : True"
    )
    print(
        "Tüm çıktılar sonlu        : True"
    )
    print(
        "Quantized engine          : "
        f"{args.backend}"
    )
    print(
        "Validation kullanıldı     : False"
    )
    print(
        "Test kullanıldı           : False"
    )
    print(
        "Doğrulama geçti           : True"
    )
    print()
    print(
        f"CSV raporu      : {report_file}"
    )
    print(
        f"JSON özet       : {summary_file}"
    )
    print(
        f"Artifact klasörü: {artifact_directory}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()