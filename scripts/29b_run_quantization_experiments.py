"""
N-BaIoT family_3 ana quantization deneyleri.

Yöntemler:
- DQ  : ONNX Runtime dynamic INT8
- PTQ : ONNX Runtime statik QDQ INT8, train-only kalibrasyon
- QAT : PyTorch FX Graph Mode QAT + oneDNN

Deney matrisi:
3 model x 5 seed x 3 yöntem = 45 sonuç.

Bilimsel protokol:
- DQ ve PTQ, eşleşen validation-seçilmiş FP32 checkpointten üretilir.
- PTQ kalibrasyonu yalnızca train splitinden yapılır.
- QAT yalnızca train splitinde güncellenir.
- QAT checkpoint seçimi dönüştürülmüş INT8 modelin validation Macro F1'ına göre yapılır.
- Test, her model-yöntem-seed kombinasyonunda bir kez değerlendirilir.
- Test, eğitim, kalibrasyon, erken durdurma veya hiperparametre seçiminde kullanılmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
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
import torch.nn.functional as F
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
from torch.fx.proxy import Proxy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)
import src.models.nbaiot_models as nbaiot_models_module  # noqa: E402
from src.models.nbaiot_models import create_model  # noqa: E402
from src.training.baseline_trainer import (  # noqa: E402
    run_epoch,
    set_reproducible_environment,
)


DEFAULT_FP32_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

DEFAULT_QAT_VALIDATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_fx_qat_pipeline_validation_summary.json"
)

DEFAULT_FP32_RUN_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_runs.csv"
)

DEFAULT_FP32_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_quantization_protocol_v1.json"
)

DEFAULT_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "quantization_v1"
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

METHOD_NAMES = (
    "DQ",
    "PTQ",
    "QAT",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

TARGET_CLASSES = (
    "benign",
    "gafgyt",
    "mirai",
)

QUANTIZED_ONNX_NODE_TYPES = {
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "QLinearConv",
    "ConvInteger",
    "QGemm",
}

AGGREGATE_METRICS = (
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_mcc",
    "macro_f1_delta_vs_fp32",
)


# ==========================================================
# ARGUMENTS
# ==========================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT family_3 için 45 DQ, PTQ ve QAT ana deneyini çalıştırır."
        )
    )

    parser.add_argument(
        "--fp32-protocol-file",
        type=Path,
        default=DEFAULT_FP32_PROTOCOL_FILE,
    )
    parser.add_argument(
        "--qat-validation-file",
        type=Path,
        default=DEFAULT_QAT_VALIDATION_FILE,
    )
    parser.add_argument(
        "--fp32-run-file",
        type=Path,
        default=DEFAULT_FP32_RUN_FILE,
    )
    parser.add_argument(
        "--fp32-experiment-directory",
        type=Path,
        default=DEFAULT_FP32_EXPERIMENT_DIRECTORY,
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=DEFAULT_PROTOCOL_FILE,
    )
    parser.add_argument(
        "--experiment-directory",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIRECTORY,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onednn",
    )
    parser.add_argument(
        "--qat-learning-rate-factor",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--qat-max-epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--qat-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--qat-min-delta",
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
    )

    return parser.parse_args()


# ==========================================================
# FILE HELPERS
# ==========================================================


def json_default(value: object) -> object:
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
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} JSON ile uyumlu değil.")


def read_json(input_file: Path) -> dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {input_file}")

    with input_file.open("r", encoding="utf-8") as file_handle:
        document = json.load(file_handle)

    if not isinstance(document, dict):
        raise TypeError(f"JSON kökü sözlük değil: {input_file}")

    return document


def write_json_atomic(document: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")

    with temporary_file.open("w", encoding="utf-8") as file_handle:
        json.dump(
            document,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )

    temporary_file.replace(output_file)


def write_csv_atomic(frame: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary_file, index=False, encoding="utf-8")
    temporary_file.replace(output_file)


def save_torch_atomic(document: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    torch.save(document, temporary_file)
    temporary_file.replace(output_file)


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"SHA-256 girdisi bulunamadı: {file_path}")

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def load_torch_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_file}")

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint kökü sözlük değil: {checkpoint_file}")

    return checkpoint


# ==========================================================
# VALIDATION AND PROTOCOL
# ==========================================================


def validate_arguments(args: argparse.Namespace) -> None:
    if args.qat_learning_rate_factor <= 0:
        raise ValueError("qat-learning-rate-factor pozitif olmalıdır.")
    if args.qat_max_epochs <= 0:
        raise ValueError("qat-max-epochs pozitif olmalıdır.")
    if args.qat_patience <= 0:
        raise ValueError("qat-patience pozitif olmalıdır.")
    if args.qat_patience > args.qat_max_epochs:
        raise ValueError("qat-patience, qat-max-epochs değerinden büyük olamaz.")
    if args.qat_min_delta < 0:
        raise ValueError("qat-min-delta negatif olamaz.")
    if args.calibration_batch_size <= 0:
        raise ValueError("calibration-batch-size pozitif olmalıdır.")
    if args.calibration_batches <= 0:
        raise ValueError("calibration-batches pozitif olmalıdır.")
    if args.opset_version <= 0:
        raise ValueError("opset-version pozitif olmalıdır.")


def validate_fp32_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_version") != "1.0":
        raise ValueError("Beklenmeyen FP32 protokol sürümü.")
    if protocol.get("task") != "family_3":
        raise ValueError("FP32 protokol görevi family_3 değil.")
    if tuple(protocol["models"]) != MODEL_NAMES:
        raise ValueError("FP32 protokol model sırası uyuşmuyor.")
    if tuple(int(seed) for seed in protocol["training_seeds"]) != SEEDS:
        raise ValueError("FP32 protokol seed sırası uyuşmuyor.")
    if tuple(protocol["target_classes"]) != TARGET_CLASSES:
        raise ValueError("FP32 protokol sınıf sırası uyuşmuyor.")


def validate_qat_validation(summary: dict[str, Any], backend: str) -> None:
    if not bool(summary.get("validation_passed")):
        raise ValueError("FX-QAT motor doğrulaması başarılı değil.")
    if int(summary.get("validated_model_count", 0)) != 3:
        raise ValueError("FX-QAT doğrulamasında üç model bulunmuyor.")
    if not bool(summary.get("all_models_have_fake_quant_modules")):
        raise ValueError("FX-QAT doğrulamasında fake-quant kapsamı eksik.")
    if not bool(summary.get("all_models_have_quantized_coverage")):
        raise ValueError("FX-QAT doğrulamasında gerçek quantized kapsam eksik.")
    if str(summary.get("backend")) != backend:
        raise ValueError("FX-QAT doğrulama backend değeri uyuşmuyor.")


def create_or_validate_protocol(
    *,
    protocol_file: Path,
    fp32_protocol_file: Path,
    fp32_protocol: dict[str, Any],
    qat_validation_file: Path,
    qat_validation: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    common = fp32_protocol["common_training_configuration"]

    expected = {
        "protocol_name": "N-BaIoT family_3 DQ PTQ QAT multi-seed",
        "protocol_version": "1.0",
        "status": "locked_before_quantization_main_experiments",
        "task": "family_3",
        "models": list(MODEL_NAMES),
        "methods": list(METHOD_NAMES),
        "training_seeds": list(SEEDS),
        "planned_run_count": 45,
        "source_policy": (
            "Every quantized artifact is produced from the matching "
            "validation-selected FP32 model-seed checkpoint."
        ),
        "dynamic_quantization": {
            "runtime": "ONNX Runtime",
            "method": "dynamic_int8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration_required": False,
        },
        "post_training_static_quantization": {
            "runtime": "ONNX Runtime",
            "method": "static_QDQ_INT8",
            "quant_format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration_method": "MinMax",
            "calibration_split": "train",
            "calibration_seed": int(args.calibration_seed),
            "calibration_batch_size": int(args.calibration_batch_size),
            "calibration_batch_count": int(args.calibration_batches),
            "calibration_sample_count": int(
                args.calibration_batch_size * args.calibration_batches
            ),
        },
        "quantization_aware_training": {
            "framework": "PyTorch FX Graph Mode QAT",
            "backend": args.backend,
            "optimizer": "AdamW",
            "learning_rate_rule": "fp32_learning_rate_times_factor",
            "learning_rate_factor": float(args.qat_learning_rate_factor),
            "max_epochs": int(args.qat_max_epochs),
            "early_stopping_patience": int(args.qat_patience),
            "early_stopping_min_delta": float(args.qat_min_delta),
            "batch_size": int(common["batch_size"]),
            "weight_decay": float(common["weight_decay"]),
            "gradient_clip_norm": float(common["gradient_clip_norm"]),
            "torch_threads": int(common["torch_threads"]),
            "num_workers": int(common["num_workers"]),
            "training_split": "train",
            "selection_split": "validation",
            "selection_metric": "converted_int8_macro_f1",
            "test_used_for_selection": False,
        },
        "onnx_export": {
            "opset_version": int(args.opset_version),
            "dynamic_batch_axis": True,
        },
        "test_evaluation": {
            "evaluations_per_model_method_seed": 1,
            "test_used_for_training": False,
            "test_used_for_calibration": False,
            "test_used_for_model_selection": False,
        },
        "source_artifacts": {
            "fp32_protocol": {
                "path": str(fp32_protocol_file),
                "sha256": calculate_sha256(fp32_protocol_file),
            },
            "qat_validation": {
                "path": str(qat_validation_file),
                "sha256": calculate_sha256(qat_validation_file),
            },
        },
    }

    if protocol_file.exists():
        protocol = read_json(protocol_file)
        for key, expected_value in expected.items():
            if protocol.get(key) != expected_value:
                raise RuntimeError(
                    "Mevcut quantization protokolü beklenen ayarlarla "
                    f"uyuşmuyor: {key}"
                )
    else:
        protocol = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **expected,
        }
        write_json_atomic(protocol, protocol_file)

    return protocol, calculate_sha256(protocol_file)


def validate_source_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model_name: str,
    seed: int,
    class_names: tuple[str, ...],
) -> None:
    if str(checkpoint.get("model_name")) != model_name:
        raise RuntimeError("Kaynak checkpoint model adı uyuşmuyor.")
    if str(checkpoint.get("task_name")) != "family_3":
        raise RuntimeError("Kaynak checkpoint görevi family_3 değil.")
    if int(checkpoint.get("seed")) != seed:
        raise RuntimeError("Kaynak checkpoint seed değeri uyuşmuyor.")
    checkpoint_classes = tuple(
        str(value) for value in checkpoint.get("class_names", [])
    )
    if checkpoint_classes != class_names:
        raise RuntimeError("Kaynak checkpoint sınıf sırası uyuşmuyor.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Kaynak checkpoint içinde model_state_dict yok.")


def load_fp32_references(run_file: Path) -> pd.DataFrame:
    if not run_file.exists():
        raise FileNotFoundError(f"FP32 sonuç dosyası bulunamadı: {run_file}")

    frame = pd.read_csv(run_file)
    required = {
        "model_name",
        "seed",
        "test_macro_f1",
        "test_mcc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "FP32 sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype("int64")
    frame["test_macro_f1"] = pd.to_numeric(
        frame["test_macro_f1"], errors="raise"
    )
    frame["test_mcc"] = pd.to_numeric(frame["test_mcc"], errors="raise")

    if len(frame) != 15:
        raise ValueError("FP32 sonuç sayısı 15 değil.")
    if frame.duplicated(subset=["model_name", "seed"]).any():
        raise ValueError("FP32 sonuçlarında tekrarlanan model-seed kaydı var.")

    return frame


# ==========================================================
# MODEL AND QAT HELPERS
# ==========================================================


def create_model_from_state(
    *,
    model_name: str,
    model_state_dict: dict[str, Any],
    input_features: int,
    num_classes: int,
) -> nn.Module:
    model = create_model(
        model_name=model_name,
        input_features=input_features,
        num_classes=num_classes,
    )
    model.load_state_dict(model_state_dict, strict=True)
    return model.cpu()


def resolve_backend_config(backend_name: str) -> tuple[Any | None, str | None]:
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
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            return function(), f"{module_name}.{function_name}"
        except (ImportError, AttributeError, ModuleNotFoundError):
            continue

    return None, None


def prepare_qat_model(
    *,
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    backend_name: str,
    backend_config: Any | None,
) -> nn.Module:
    qconfig_mapping = get_default_qat_qconfig_mapping(backend_name)
    model.train()

    original_validator = nbaiot_models_module.validate_forward_input

    def fx_safe_validate_forward_input(
        *validator_args: Any,
        **validator_kwargs: Any,
    ) -> Any:
        if "inputs" in validator_kwargs:
            input_value = validator_kwargs["inputs"]
        elif validator_args:
            input_value = validator_args[0]
        else:
            return original_validator(*validator_args, **validator_kwargs)

        if isinstance(input_value, Proxy):
            return None

        return original_validator(*validator_args, **validator_kwargs)

    nbaiot_models_module.validate_forward_input = fx_safe_validate_forward_input

    try:
        if backend_config is None:
            prepared = prepare_qat_fx(
                model,
                qconfig_mapping,
                example_inputs,
            )
        else:
            prepared = prepare_qat_fx(
                model,
                qconfig_mapping,
                example_inputs,
                backend_config=backend_config,
            )
    finally:
        nbaiot_models_module.validate_forward_input = original_validator

    return prepared


def convert_qat_model(
    *,
    prepared_model: nn.Module,
    backend_config: Any | None,
) -> nn.Module:
    prepared_model.eval()

    if backend_config is None:
        converted = convert_fx(prepared_model)
    else:
        converted = convert_fx(
            prepared_model,
            backend_config=backend_config,
        )

    converted.eval()
    return converted


def create_prepared_snapshot(
    *,
    model_name: str,
    source_model_state_dict: dict[str, Any],
    prepared_state_dict: dict[str, Any],
    input_features: int,
    num_classes: int,
    example_inputs: tuple[torch.Tensor, ...],
    backend_name: str,
    backend_config: Any | None,
) -> nn.Module:
    source_model = create_model_from_state(
        model_name=model_name,
        model_state_dict=source_model_state_dict,
        input_features=input_features,
        num_classes=num_classes,
    )

    prepared = prepare_qat_model(
        model=source_model,
        example_inputs=example_inputs,
        backend_name=backend_name,
        backend_config=backend_config,
    )
    prepared.load_state_dict(prepared_state_dict, strict=True)
    return prepared


def count_fake_quant_modules(model: nn.Module) -> int:
    return int(
        sum(isinstance(module, FakeQuantizeBase) for module in model.modules())
    )


def count_quantized_modules(model: nn.Module) -> tuple[int, list[str]]:
    module_types: list[str] = []

    for module in model.modules():
        module_path = type(module).__module__.lower()
        if ".quantized" in module_path or ".intrinsic.quantized" in module_path:
            module_types.append(
                f"{type(module).__module__}.{type(module).__name__}"
            )

    return len(module_types), sorted(set(module_types))


def count_quantized_graph_nodes(model: nn.Module) -> tuple[int, list[str]]:
    graph = getattr(model, "graph", None)
    if graph is None:
        return 0, []

    targets: list[str] = []

    for node in graph.nodes:
        target_text = str(node.target).lower()
        node_text = str(node).lower()

        if (
            "quantized" in target_text
            or "quantize_per_tensor" in target_text
            or "dequantize" in target_text
            or "quantized" in node_text
        ):
            targets.append(str(node.target))

    return len(targets), sorted(set(targets))


# ==========================================================
# ONNX HELPERS
# ==========================================================


def export_model_to_onnx(
    *,
    model: nn.Module,
    sample_features: torch.Tensor,
    output_file: Path,
    opset_version: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    sample_features = sample_features.detach().cpu().float()

    export_arguments = {
        "model": model,
        "args": (sample_features,),
        "f": str(output_file),
        "input_names": ["features"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        "opset_version": int(opset_version),
        "do_constant_folding": True,
        "export_params": True,
    }

    try:
        torch.onnx.export(**export_arguments, dynamo=False)
    except TypeError:
        torch.onnx.export(**export_arguments)

    if not output_file.exists() or output_file.stat().st_size <= 0:
        raise RuntimeError("ONNX dışa aktarımı geçerli dosya oluşturmadı.")


def validate_onnx_model(model_file: Path) -> None:
    import onnx

    model = onnx.load(str(model_file))
    onnx.checker.check_model(model)


def create_ort_session(
    *,
    model_file: Path,
    torch_threads: int,
) -> Any:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(torch_threads)
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def compare_pytorch_onnx(
    *,
    model: nn.Module,
    onnx_file: Path,
    features: torch.Tensor,
    torch_threads: int,
) -> dict[str, Any]:
    model.eval()

    with torch.inference_mode():
        pytorch_logits = model(features).detach().cpu().float().numpy()

    session = create_ort_session(
        model_file=onnx_file,
        torch_threads=torch_threads,
    )
    input_name = session.get_inputs()[0].name
    onnx_logits = np.asarray(
        session.run(
            None,
            {
                input_name: np.ascontiguousarray(
                    features.detach().cpu().numpy().astype(np.float32, copy=False)
                )
            },
        )[0],
        dtype=np.float32,
    )

    if pytorch_logits.shape != onnx_logits.shape:
        raise RuntimeError("PyTorch ve ONNX logit boyutları uyuşmuyor.")

    differences = np.abs(pytorch_logits - onnx_logits)
    allclose = bool(
        np.allclose(
            pytorch_logits,
            onnx_logits,
            rtol=1e-4,
            atol=1e-5,
        )
    )

    if not allclose:
        raise RuntimeError("PyTorch ve FP32 ONNX çıktıları tolerans içinde değil.")

    return {
        "allclose": True,
        "maximum_absolute_difference": float(differences.max()),
        "mean_absolute_difference": float(differences.mean()),
        "argmax_prediction_agreement": float(
            np.mean(
                np.argmax(pytorch_logits, axis=1)
                == np.argmax(onnx_logits, axis=1)
            )
        ),
    }


def summarize_onnx_graph(model_file: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(model_file))
    node_counts = Counter(node.op_type for node in model.graph.node)
    quantized_node_count = int(
        sum(node_counts.get(node_type, 0) for node_type in QUANTIZED_ONNX_NODE_TYPES)
    )

    return {
        "node_count": int(len(model.graph.node)),
        "node_type_counts": dict(sorted(node_counts.items())),
        "quantized_node_count": quantized_node_count,
        "quantized_node_types_present": sorted(
            node_type
            for node_type in QUANTIZED_ONNX_NODE_TYPES
            if node_counts.get(node_type, 0) > 0
        ),
    }


def call_dynamic_quantization(
    *,
    input_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_file.parent.mkdir(parents=True, exist_ok=True)

    call_arguments: dict[str, Any] = {
        "model_input": str(input_file),
        "model_output": str(output_file),
        "weight_type": QuantType.QInt8,
        "per_channel": True,
        "reduce_range": False,
    }

    signature = inspect.signature(quantize_dynamic)
    accepted = {
        key: value
        for key, value in call_arguments.items()
        if key in signature.parameters
    }
    quantize_dynamic(**accepted)

    if not output_file.exists() or output_file.stat().st_size <= 0:
        raise RuntimeError("Dynamic quantization geçerli dosya oluşturmadı.")

    return {
        "function_signature": str(signature),
        "used_arguments": {
            key: str(value)
            for key, value in accepted.items()
            if key not in {"model_input", "model_output"}
        },
    }


def create_calibration_reader(
    *,
    input_name: str,
    batches: list[np.ndarray],
) -> Any:
    from onnxruntime.quantization import CalibrationDataReader

    class TrainOnlyCalibrationReader(CalibrationDataReader):
        def __init__(
            self,
            model_input_name: str,
            calibration_batches: list[np.ndarray],
        ) -> None:
            self.model_input_name = model_input_name
            self.calibration_batches = [
                np.ascontiguousarray(batch.astype(np.float32, copy=False))
                for batch in calibration_batches
            ]
            self.rewind()

        def get_next(self) -> dict[str, np.ndarray] | None:
            try:
                batch = next(self._iterator)
            except StopIteration:
                return None

            return {self.model_input_name: batch}

        def rewind(self) -> None:
            self._iterator = iter(self.calibration_batches)

    return TrainOnlyCalibrationReader(input_name, batches)


def call_static_quantization(
    *,
    input_file: Path,
    output_file: Path,
    calibration_batches: list[np.ndarray],
    input_name: str,
) -> dict[str, Any]:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    reader = create_calibration_reader(
        input_name=input_name,
        batches=calibration_batches,
    )

    call_arguments: dict[str, Any] = {
        "model_input": str(input_file),
        "model_output": str(output_file),
        "calibration_data_reader": reader,
        "quant_format": QuantFormat.QDQ,
        "activation_type": QuantType.QUInt8,
        "weight_type": QuantType.QInt8,
        "per_channel": True,
        "reduce_range": False,
        "calibrate_method": CalibrationMethod.MinMax,
    }

    signature = inspect.signature(quantize_static)
    accepted = {
        key: value
        for key, value in call_arguments.items()
        if key in signature.parameters
    }
    quantize_static(**accepted)

    if not output_file.exists() or output_file.stat().st_size <= 0:
        raise RuntimeError("Statik PTQ geçerli dosya oluşturmadı.")

    return {
        "function_signature": str(signature),
        "calibration_split": "train",
        "calibration_batch_count": int(len(calibration_batches)),
        "calibration_sample_count": int(sum(len(batch) for batch in calibration_batches)),
        "used_arguments": {
            key: str(value)
            for key, value in accepted.items()
            if key not in {
                "model_input",
                "model_output",
                "calibration_data_reader",
            }
        },
    }


def obtain_calibration_batches(
    *,
    assets: Any,
    batch_size: int,
    batch_count: int,
    seed: int,
) -> list[np.ndarray]:
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

    batches: list[np.ndarray] = []

    for features, _ in train_loader:
        batch = np.array(
            features.detach().cpu().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )
        batches.append(batch)

        if len(batches) >= batch_count:
            break

    if len(batches) != batch_count:
        raise RuntimeError("İstenen sayıda kalibrasyon batchi alınamadı.")

    return batches


# ==========================================================
# METRICS
# ==========================================================


def build_metrics_from_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average_loss: float,
    class_names: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise RuntimeError("Gerçek ve tahmin etiket boyutları uyuşmuyor.")

    labels = np.arange(len(class_names), dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_class: dict[str, dict[str, Any]] = {}

    for class_index, class_name in enumerate(class_names):
        class_support = int(support[class_index])
        predicted_count = int(np.sum(y_pred == class_index))
        class_recall = float(recall[class_index])

        per_class[class_name] = {
            "support": class_support,
            "predicted_count": predicted_count,
            "precision": float(precision[class_index]),
            "recall": class_recall,
            "f1": float(f1[class_index]),
            "false_negative_rate": float(1.0 - class_recall),
        }

    sample_count = int(len(y_true))

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "sample_count": sample_count,
        "elapsed_seconds": float(elapsed_seconds),
        "throughput_samples_per_second": float(
            sample_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "per_class": per_class,
    }


def evaluate_onnx_split(
    *,
    model_file: Path,
    assets: Any,
    split_name: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    torch_threads: int,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    session = create_ort_session(
        model_file=model_file,
        torch_threads=torch_threads,
    )
    input_name = session.get_inputs()[0].name

    _, data_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name=split_name,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=False,
    )

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    total_weighted_loss = 0.0
    total_target_weight = 0.0
    total_samples = 0
    start_time = time.perf_counter()

    for features, targets in data_loader:
        feature_array = np.array(
            features.detach().cpu().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )

        logits = np.asarray(
            session.run(None, {input_name: feature_array})[0],
            dtype=np.float32,
        )

        if not np.isfinite(logits).all():
            raise RuntimeError("ONNX Runtime sonlu olmayan logit üretti.")

        target_cpu = targets.detach().cpu().long()
        batch_loss = F.cross_entropy(
            torch.from_numpy(logits),
            target_cpu,
            weight=class_weights,
            reduction="none",
        )

        total_weighted_loss += float(batch_loss.double().sum().item())
        total_target_weight += float(
            class_weights[target_cpu].double().sum().item()
        )
        total_samples += int(len(target_cpu))
        y_true_parts.append(target_cpu.numpy().astype(np.int64, copy=False))
        y_pred_parts.append(np.argmax(logits, axis=1).astype(np.int64, copy=False))

    elapsed_seconds = float(time.perf_counter() - start_time)

    if total_samples <= 0:
        raise RuntimeError(f"{split_name} değerlendirmesinde örnek bulunamadı.")

    if total_target_weight <= 0.0:
        raise RuntimeError("Ağırlıklı loss paydası pozitif değil.")

    return build_metrics_from_predictions(
        y_true=np.concatenate(y_true_parts),
        y_pred=np.concatenate(y_pred_parts),
        average_loss=total_weighted_loss / total_target_weight,
        class_names=tuple(assets.target_classes),
        elapsed_seconds=elapsed_seconds,
    )


def calculate_summary_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("İstatistik dizisi tek boyutlu ve boş olmayan olmalıdır.")
    if not np.isfinite(values).all():
        raise ValueError("İstatistik dizisinde sonlu olmayan değer var.")

    count = int(len(values))
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
    standard_error = float(standard_deviation / np.sqrt(count)) if count > 1 else 0.0
    critical = float(student_t.ppf(0.975, df=count - 1)) if count > 1 else 0.0
    margin = float(critical * standard_error)

    return {
        "sample_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "ci95_lower": float(mean - margin),
        "ci95_upper": float(mean + margin),
    }


# ==========================================================
# RESULT WRITERS
# ==========================================================


def write_test_artifacts(
    *,
    metrics: dict[str, Any],
    run_directory: Path,
    method_name: str,
    model_name: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    test_metrics_file = run_directory / "test_metrics.json"
    per_class_file = run_directory / "test_per_class_metrics.csv"
    confusion_file = run_directory / "test_confusion_matrix.csv"

    write_json_atomic(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "method_name": method_name,
            "seed": int(seed),
            "task_name": "family_3",
            "test_evaluation_attempt_count": 1,
            "test_used_for_selection": False,
            **metrics,
        },
        test_metrics_file,
    )

    per_class_records: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(TARGET_CLASSES):
        class_metrics = metrics["per_class"][class_name]
        per_class_records.append(
            {
                "class_index": int(class_index),
                "class_name": class_name,
                "support": int(class_metrics["support"]),
                "predicted_count": int(class_metrics["predicted_count"]),
                "precision": float(class_metrics["precision"]),
                "recall": float(class_metrics["recall"]),
                "f1": float(class_metrics["f1"]),
                "false_negative_rate": float(
                    class_metrics["false_negative_rate"]
                ),
            }
        )

    write_csv_atomic(pd.DataFrame(per_class_records), per_class_file)

    confusion_frame = pd.DataFrame(
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        index=TARGET_CLASSES,
        columns=TARGET_CLASSES,
    )
    confusion_frame.index.name = "true_class"
    confusion_frame.to_csv(confusion_file, encoding="utf-8")

    return test_metrics_file, per_class_file, confusion_file


def begin_test_manifest(
    *,
    run_directory: Path,
    model_name: str,
    method_name: str,
    seed: int,
    artifact_file: Path,
    restart_incomplete: bool,
) -> Path:
    marker_file = run_directory / "test_evaluation_manifest.json"

    if marker_file.exists():
        marker = read_json(marker_file)
        status = marker.get("status")

        if status == "completed":
            return marker_file

        if status == "started" and restart_incomplete:
            marker_file.unlink()
        else:
            raise RuntimeError(
                "Yarım kalmış test değerlendirme manifesti bulundu: "
                f"{marker_file}\nYeniden başlatmak için --restart-incomplete kullan."
            )

    write_json_atomic(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "model_name": model_name,
            "method_name": method_name,
            "seed": int(seed),
            "artifact_file": str(artifact_file),
            "artifact_sha256": calculate_sha256(artifact_file),
            "evaluation_attempt_count": 1,
            "test_used_for_selection": False,
        },
        marker_file,
    )

    return marker_file


def complete_test_manifest(
    *,
    marker_file: Path,
    test_metrics_file: Path,
    per_class_file: Path,
    confusion_file: Path,
) -> None:
    marker = read_json(marker_file)
    marker.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "test_metrics_file": str(test_metrics_file),
            "test_per_class_file": str(per_class_file),
            "test_confusion_file": str(confusion_file),
        }
    )
    write_json_atomic(marker, marker_file)


# ==========================================================
# QAT TRAINING
# ==========================================================


def train_qat_model(
    *,
    model_name: str,
    seed: int,
    source_model_state_dict: dict[str, Any],
    source_checkpoint_file: Path,
    assets: Any,
    fp32_learning_rate: float,
    protocol: dict[str, Any],
    protocol_sha256: str,
    backend_config: Any | None,
    backend_config_source: str | None,
    run_directory: Path,
) -> dict[str, Any]:
    qat_config = protocol["quantization_aware_training"]
    run_directory.mkdir(parents=True, exist_ok=True)

    training_summary_file = run_directory / "training_summary.json"
    history_file = run_directory / "training_history.csv"
    best_prepared_state_file = run_directory / "best_prepared_qat_state_dict.pt"

    if training_summary_file.exists():
        return read_json(training_summary_file)

    set_reproducible_environment(
        seed=seed,
        torch_threads=int(qat_config["torch_threads"]),
    )
    torch.backends.quantized.engine = str(qat_config["backend"])

    _, example_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="train",
        batch_size=8,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=0,
        pin_memory=False,
    )
    example_features, _ = next(iter(example_loader))
    example_inputs = (example_features.cpu().contiguous(),)

    source_model = create_model_from_state(
        model_name=model_name,
        model_state_dict=source_model_state_dict,
        input_features=assets.feature_count,
        num_classes=assets.class_count,
    )

    prepared_model = prepare_qat_model(
        model=source_model,
        example_inputs=example_inputs,
        backend_name=str(qat_config["backend"]),
        backend_config=backend_config,
    )

    fake_quant_count = count_fake_quant_modules(prepared_model)
    if fake_quant_count <= 0:
        raise RuntimeError("QAT modeli fake-quant modülü içermiyor.")

    class_weights = torch.as_tensor(assets.class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    qat_learning_rate = float(
        fp32_learning_rate * float(qat_config["learning_rate_factor"])
    )
    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=qat_learning_rate,
        weight_decay=float(qat_config["weight_decay"]),
    )

    best_validation_macro_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_records: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    for epoch in range(1, int(qat_config["max_epochs"]) + 1):
        _, train_loader = create_nbaiot_dataloader(
            assets=assets,
            split_name="train",
            batch_size=int(qat_config["batch_size"]),
            shuffle=True,
            seed=seed,
            epoch=epoch,
            num_workers=int(qat_config["num_workers"]),
            pin_memory=False,
        )

        train_metrics = run_epoch(
            model=prepared_model,
            data_loader=train_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=optimizer,
            gradient_clip_norm=float(qat_config["gradient_clip_norm"]),
            maximum_batches=None,
        )

        prepared_snapshot = create_prepared_snapshot(
            model_name=model_name,
            source_model_state_dict=source_model_state_dict,
            prepared_state_dict=prepared_model.state_dict(),
            input_features=assets.feature_count,
            num_classes=assets.class_count,
            example_inputs=example_inputs,
            backend_name=str(qat_config["backend"]),
            backend_config=backend_config,
        )
        converted_validation_model = convert_qat_model(
            prepared_model=prepared_snapshot,
            backend_config=backend_config,
        )

        quantized_module_count, _ = count_quantized_modules(
            converted_validation_model
        )
        quantized_node_count, _ = count_quantized_graph_nodes(
            converted_validation_model
        )
        if quantized_module_count + quantized_node_count <= 0:
            raise RuntimeError("QAT validation modeli quantized kapsam içermiyor.")

        _, validation_loader = create_nbaiot_dataloader(
            assets=assets,
            split_name="validation",
            batch_size=int(qat_config["batch_size"]),
            shuffle=False,
            seed=seed,
            epoch=0,
            num_workers=int(qat_config["num_workers"]),
            pin_memory=False,
        )

        validation_metrics = run_epoch(
            model=converted_validation_model,
            data_loader=validation_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=None,
            gradient_clip_norm=0.0,
            maximum_batches=None,
        )

        validation_macro_f1 = float(validation_metrics["macro_f1"])
        improved = bool(
            validation_macro_f1
            > best_validation_macro_f1 + float(qat_config["early_stopping_min_delta"])
        )

        if improved:
            best_validation_macro_f1 = validation_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_atomic(prepared_model.state_dict(), best_prepared_state_file)
        else:
            epochs_without_improvement += 1

        history_records.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "train_mcc": float(
                    train_metrics["matthews_correlation_coefficient"]
                ),
                "validation_loss_converted_int8": float(
                    validation_metrics["loss"]
                ),
                "validation_accuracy_converted_int8": float(
                    validation_metrics["accuracy"]
                ),
                "validation_macro_f1_converted_int8": validation_macro_f1,
                "validation_mcc_converted_int8": float(
                    validation_metrics["matthews_correlation_coefficient"]
                ),
                "improved": improved,
                "epochs_without_improvement": int(epochs_without_improvement),
            }
        )

        print(
            f"    epoch={epoch:02d} | "
            f"train_macro_f1={train_metrics['macro_f1']:.6f} | "
            f"val_int8_macro_f1={validation_macro_f1:.6f} | "
            f"best={best_validation_macro_f1:.6f} | "
            f"improved={improved}"
        )

        if epochs_without_improvement >= int(
            qat_config["early_stopping_patience"]
        ):
            break

    total_training_seconds = float(time.perf_counter() - training_start)

    if best_epoch <= 0 or not best_prepared_state_file.exists():
        raise RuntimeError("QAT en iyi prepared state oluşturmadı.")

    write_csv_atomic(pd.DataFrame(history_records), history_file)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "method_name": "QAT",
        "seed": int(seed),
        "source_checkpoint_file": str(source_checkpoint_file),
        "source_checkpoint_sha256": calculate_sha256(source_checkpoint_file),
        "quantization_protocol_sha256": protocol_sha256,
        "backend": str(qat_config["backend"]),
        "backend_config_source": backend_config_source,
        "fp32_learning_rate": float(fp32_learning_rate),
        "qat_learning_rate": qat_learning_rate,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_records)),
        "best_validation_macro_f1_converted_int8": float(
            best_validation_macro_f1
        ),
        "fake_quant_module_count": int(fake_quant_count),
        "total_training_seconds": total_training_seconds,
        "best_prepared_state_file": str(best_prepared_state_file),
        "best_prepared_state_sha256": calculate_sha256(best_prepared_state_file),
        "history_file": str(history_file),
        "training_split": "train",
        "selection_split": "validation",
        "selection_metric": "converted_int8_macro_f1",
        "test_split_evaluated": False,
        "training_completed": True,
    }

    write_json_atomic(summary, training_summary_file)
    return summary


def evaluate_qat_test_once(
    *,
    model_name: str,
    seed: int,
    source_model_state_dict: dict[str, Any],
    assets: Any,
    protocol: dict[str, Any],
    backend_config: Any | None,
    training_summary: dict[str, Any],
    run_directory: Path,
    restart_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    qat_config = protocol["quantization_aware_training"]
    test_metrics_file = run_directory / "test_metrics.json"
    converted_state_file = run_directory / "converted_int8_state_dict.pt"

    if test_metrics_file.exists():
        metrics_document = read_json(test_metrics_file)
        metrics = {
            key: value
            for key, value in metrics_document.items()
            if key
            not in {
                "generated_at_utc",
                "model_name",
                "method_name",
                "seed",
                "task_name",
                "test_evaluation_attempt_count",
                "test_used_for_selection",
            }
        }
        artifact_metadata = read_json(run_directory / "artifact_metadata.json")
        return metrics, artifact_metadata

    _, example_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="train",
        batch_size=8,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=0,
        pin_memory=False,
    )
    example_features, _ = next(iter(example_loader))
    example_inputs = (example_features.cpu().contiguous(),)

    prepared_state = load_torch_checkpoint(
        Path(training_summary["best_prepared_state_file"])
    )

    prepared_model = create_prepared_snapshot(
        model_name=model_name,
        source_model_state_dict=source_model_state_dict,
        prepared_state_dict=prepared_state,
        input_features=assets.feature_count,
        num_classes=assets.class_count,
        example_inputs=example_inputs,
        backend_name=str(qat_config["backend"]),
        backend_config=backend_config,
    )
    converted_model = convert_qat_model(
        prepared_model=prepared_model,
        backend_config=backend_config,
    )

    quantized_module_count, quantized_module_types = count_quantized_modules(
        converted_model
    )
    quantized_node_count, quantized_node_targets = count_quantized_graph_nodes(
        converted_model
    )

    if quantized_module_count + quantized_node_count <= 0:
        raise RuntimeError("QAT test modeli quantized kapsam içermiyor.")

    save_torch_atomic(converted_model.state_dict(), converted_state_file)
    graph_file = run_directory / "converted_fx_graph.txt"
    graph_file.write_text(str(converted_model.graph), encoding="utf-8")

    marker_file = begin_test_manifest(
        run_directory=run_directory,
        model_name=model_name,
        method_name="QAT",
        seed=seed,
        artifact_file=converted_state_file,
        restart_incomplete=restart_incomplete,
    )

    class_weights = torch.as_tensor(assets.class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    _, test_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="test",
        batch_size=int(qat_config["batch_size"]),
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=int(qat_config["num_workers"]),
        pin_memory=False,
    )

    metrics = run_epoch(
        model=converted_model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=assets.target_classes,
        optimizer=None,
        gradient_clip_norm=0.0,
        maximum_batches=None,
    )

    test_metrics_file, per_class_file, confusion_file = write_test_artifacts(
        metrics=metrics,
        run_directory=run_directory,
        method_name="QAT",
        model_name=model_name,
        seed=seed,
    )
    complete_test_manifest(
        marker_file=marker_file,
        test_metrics_file=test_metrics_file,
        per_class_file=per_class_file,
        confusion_file=confusion_file,
    )

    artifact_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_name": "QAT",
        "artifact_file": str(converted_state_file),
        "artifact_sha256": calculate_sha256(converted_state_file),
        "artifact_size_bytes": int(converted_state_file.stat().st_size),
        "prepared_state_file": str(training_summary["best_prepared_state_file"]),
        "prepared_state_sha256": calculate_sha256(
            Path(training_summary["best_prepared_state_file"])
        ),
        "quantized_module_count": int(quantized_module_count),
        "quantized_module_types": quantized_module_types,
        "quantized_graph_node_count": int(quantized_node_count),
        "quantized_graph_targets": quantized_node_targets,
        "graph_file": str(graph_file),
        "native_artifact_note": (
            "PyTorch quantized state_dict size is runtime-specific and is not "
            "directly comparable with ONNX file size."
        ),
    }
    write_json_atomic(artifact_metadata, run_directory / "artifact_metadata.json")

    return metrics, artifact_metadata


# ==========================================================
# DQ / PTQ
# ==========================================================


def run_onnx_method(
    *,
    method_name: str,
    model_name: str,
    seed: int,
    fp32_onnx_file: Path,
    run_directory: Path,
    assets: Any,
    protocol: dict[str, Any],
    calibration_batches: list[np.ndarray],
    restart_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    test_metrics_file = run_directory / "test_metrics.json"
    artifact_metadata_file = run_directory / "artifact_metadata.json"

    if test_metrics_file.exists() and artifact_metadata_file.exists():
        metrics_document = read_json(test_metrics_file)
        metrics = {
            key: value
            for key, value in metrics_document.items()
            if key
            not in {
                "generated_at_utc",
                "model_name",
                "method_name",
                "seed",
                "task_name",
                "test_evaluation_attempt_count",
                "test_used_for_selection",
            }
        }
        return metrics, read_json(artifact_metadata_file)

    run_directory.mkdir(parents=True, exist_ok=True)

    if method_name == "DQ":
        artifact_file = run_directory / "model_dynamic_int8.onnx"
        conversion_details = call_dynamic_quantization(
            input_file=fp32_onnx_file,
            output_file=artifact_file,
        )
    elif method_name == "PTQ":
        artifact_file = run_directory / "model_static_ptq_int8.onnx"
        fp32_session = create_ort_session(
            model_file=fp32_onnx_file,
            torch_threads=int(
                protocol["quantization_aware_training"]["torch_threads"]
            ),
        )
        input_name = fp32_session.get_inputs()[0].name
        conversion_details = call_static_quantization(
            input_file=fp32_onnx_file,
            output_file=artifact_file,
            calibration_batches=calibration_batches,
            input_name=input_name,
        )
    else:
        raise ValueError(f"Desteklenmeyen ONNX yöntemi: {method_name}")

    validate_onnx_model(artifact_file)
    graph_summary = summarize_onnx_graph(artifact_file)

    if int(graph_summary["quantized_node_count"]) <= 0:
        raise RuntimeError(
            f"{model_name}/{method_name}/seed{seed}: quantized ONNX düğümü yok."
        )

    marker_file = begin_test_manifest(
        run_directory=run_directory,
        model_name=model_name,
        method_name=method_name,
        seed=seed,
        artifact_file=artifact_file,
        restart_incomplete=restart_incomplete,
    )

    qat_config = protocol["quantization_aware_training"]
    class_weights = torch.as_tensor(assets.class_weights, dtype=torch.float32)
    metrics = evaluate_onnx_split(
        model_file=artifact_file,
        assets=assets,
        split_name="test",
        batch_size=int(qat_config["batch_size"]),
        seed=seed,
        num_workers=int(qat_config["num_workers"]),
        torch_threads=int(qat_config["torch_threads"]),
        class_weights=class_weights,
    )

    test_metrics_file, per_class_file, confusion_file = write_test_artifacts(
        metrics=metrics,
        run_directory=run_directory,
        method_name=method_name,
        model_name=model_name,
        seed=seed,
    )
    complete_test_manifest(
        marker_file=marker_file,
        test_metrics_file=test_metrics_file,
        per_class_file=per_class_file,
        confusion_file=confusion_file,
    )

    artifact_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_name": method_name,
        "artifact_file": str(artifact_file),
        "artifact_sha256": calculate_sha256(artifact_file),
        "artifact_size_bytes": int(artifact_file.stat().st_size),
        "fp32_onnx_file": str(fp32_onnx_file),
        "fp32_onnx_sha256": calculate_sha256(fp32_onnx_file),
        "fp32_onnx_size_bytes": int(fp32_onnx_file.stat().st_size),
        "size_ratio_vs_fp32_onnx": float(
            artifact_file.stat().st_size / fp32_onnx_file.stat().st_size
        ),
        "graph_summary": graph_summary,
        "conversion_details": conversion_details,
    }
    write_json_atomic(artifact_metadata, artifact_metadata_file)

    return metrics, artifact_metadata


# ==========================================================
# AGGREGATION
# ==========================================================


def aggregate_results(run_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        for method_name in METHOD_NAMES:
            subset = run_frame[
                (run_frame["model_name"] == model_name)
                & (run_frame["method_name"] == method_name)
            ]

            for metric_name in AGGREGATE_METRICS:
                statistics = calculate_summary_statistics(
                    subset[metric_name].to_numpy(dtype=np.float64)
                )
                records.append(
                    {
                        "model_name": model_name,
                        "method_name": method_name,
                        "metric": metric_name,
                        **statistics,
                    }
                )

    return pd.DataFrame(records)


def aggregate_per_class(per_class_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        for method_name in METHOD_NAMES:
            for class_name in TARGET_CLASSES:
                subset = per_class_frame[
                    (per_class_frame["model_name"] == model_name)
                    & (per_class_frame["method_name"] == method_name)
                    & (per_class_frame["class_name"] == class_name)
                ]

                for metric_name in (
                    "precision",
                    "recall",
                    "f1",
                    "false_negative_rate",
                ):
                    statistics = calculate_summary_statistics(
                        subset[metric_name].to_numpy(dtype=np.float64)
                    )
                    records.append(
                        {
                            "model_name": model_name,
                            "method_name": method_name,
                            "class_name": class_name,
                            "metric": metric_name,
                            **statistics,
                        }
                    )

    return pd.DataFrame(records)


# ==========================================================
# MAIN
# ==========================================================


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    fp32_protocol_file = args.fp32_protocol_file.resolve()
    qat_validation_file = args.qat_validation_file.resolve()
    fp32_run_file = args.fp32_run_file.resolve()
    fp32_experiment_directory = args.fp32_experiment_directory.resolve()
    protocol_file = args.protocol_file.resolve()
    experiment_directory = args.experiment_directory.resolve()
    report_directory = args.report_directory.resolve()

    experiment_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    protocol_file.parent.mkdir(parents=True, exist_ok=True)

    fp32_protocol = read_json(fp32_protocol_file)
    validate_fp32_protocol(fp32_protocol)

    qat_validation = read_json(qat_validation_file)
    validate_qat_validation(qat_validation, args.backend)

    supported_engines = tuple(torch.backends.quantized.supported_engines)
    if args.backend not in supported_engines:
        raise RuntimeError(
            f"{args.backend} desteklenmiyor. Desteklenenler={supported_engines}"
        )
    torch.backends.quantized.engine = args.backend

    protocol, protocol_sha256 = create_or_validate_protocol(
        protocol_file=protocol_file,
        fp32_protocol_file=fp32_protocol_file,
        fp32_protocol=fp32_protocol,
        qat_validation_file=qat_validation_file,
        qat_validation=qat_validation,
        args=args,
    )

    fp32_references = load_fp32_references(fp32_run_file)

    assets = load_pipeline_assets(
        task_name="family_3",
        weight_scheme="inverse_square_root_frequency_mean1",
        expected_seed=2026,
    )

    if assets.feature_count != 115:
        raise RuntimeError("Beklenen özellik sayısı 115 değil.")
    if tuple(assets.target_classes) != TARGET_CLASSES:
        raise RuntimeError("family_3 sınıf sırası uyuşmuyor.")

    calibration_batches = obtain_calibration_batches(
        assets=assets,
        batch_size=int(args.calibration_batch_size),
        batch_count=int(args.calibration_batches),
        seed=int(args.calibration_seed),
    )

    common = fp32_protocol["common_training_configuration"]
    torch_threads = int(common["torch_threads"])
    torch.set_num_threads(torch_threads)

    backend_config, backend_config_source = resolve_backend_config(args.backend)

    run_records: list[dict[str, Any]] = []
    per_class_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []

    total_run_count = len(MODEL_NAMES) * len(METHOD_NAMES) * len(SEEDS)
    run_index = 0

    print("=" * 78)
    print("N-BaIoT Family-3 DQ, PTQ ve QAT Ana Deneyleri")
    print("=" * 78)
    print("Modeller      : " + ", ".join(MODEL_NAMES))
    print("Yöntemler     : " + ", ".join(METHOD_NAMES))
    print("Seedler       : " + ", ".join(str(seed) for seed in SEEDS))
    print(f"Toplam sonuç  : {total_run_count}")
    print(f"QAT backend   : {args.backend}")
    print(
        "PTQ kalibrasyon: "
        f"{args.calibration_batches} x {args.calibration_batch_size} "
        "train örneği batchi"
    )
    print("QAT seçimi    : converted INT8 validation Macro F1")
    print("Test politikası: her model-yöntem-seed için bir kez")
    print(f"Protokol SHA  : {protocol_sha256}")
    print("=" * 78)

    for model_name in MODEL_NAMES:
        fp32_learning_rate = float(
            fp32_protocol["model_specific_configuration"][model_name][
                "learning_rate"
            ]
        )

        for seed in SEEDS:
            source_checkpoint_file = (
                fp32_experiment_directory
                / model_name
                / f"seed{seed}"
                / "best_checkpoint.pt"
            )
            source_checkpoint = load_torch_checkpoint(source_checkpoint_file)
            validate_source_checkpoint(
                checkpoint=source_checkpoint,
                model_name=model_name,
                seed=seed,
                class_names=assets.target_classes,
            )
            source_state = source_checkpoint["model_state_dict"]

            shared_directory = (
                experiment_directory / model_name / "shared" / f"seed{seed}"
            )
            shared_directory.mkdir(parents=True, exist_ok=True)
            fp32_onnx_file = shared_directory / "model_fp32.onnx"

            _, example_loader = create_nbaiot_dataloader(
                assets=assets,
                split_name="train",
                batch_size=64,
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=0,
                pin_memory=False,
            )
            example_features, _ = next(iter(example_loader))

            fp32_model = create_model_from_state(
                model_name=model_name,
                model_state_dict=source_state,
                input_features=assets.feature_count,
                num_classes=assets.class_count,
            )

            if not fp32_onnx_file.exists():
                export_model_to_onnx(
                    model=fp32_model,
                    sample_features=example_features[:8],
                    output_file=fp32_onnx_file,
                    opset_version=int(protocol["onnx_export"]["opset_version"]),
                )
                validate_onnx_model(fp32_onnx_file)
                parity = compare_pytorch_onnx(
                    model=fp32_model,
                    onnx_file=fp32_onnx_file,
                    features=example_features,
                    torch_threads=torch_threads,
                )
                write_json_atomic(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model_name": model_name,
                        "seed": int(seed),
                        "source_checkpoint_file": str(source_checkpoint_file),
                        "source_checkpoint_sha256": calculate_sha256(
                            source_checkpoint_file
                        ),
                        "fp32_onnx_file": str(fp32_onnx_file),
                        "fp32_onnx_sha256": calculate_sha256(fp32_onnx_file),
                        "fp32_onnx_size_bytes": int(fp32_onnx_file.stat().st_size),
                        "pytorch_onnx_parity": parity,
                    },
                    shared_directory / "fp32_onnx_metadata.json",
                )
            else:
                validate_onnx_model(fp32_onnx_file)

            fp32_reference = fp32_references[
                (fp32_references["model_name"] == model_name)
                & (fp32_references["seed"] == seed)
            ].iloc[0]

            for method_name in METHOD_NAMES:
                run_index += 1
                run_directory = (
                    experiment_directory
                    / model_name
                    / method_name
                    / f"seed{seed}"
                )
                run_directory.mkdir(parents=True, exist_ok=True)
                final_summary_file = run_directory / "run_summary.json"

                print()
                print("-" * 78)
                print(f"Çalışma {run_index}/{total_run_count}")
                print(f"Model  : {model_name}")
                print(f"Yöntem : {method_name}")
                print(f"Seed   : {seed}")
                print("-" * 78)

                if final_summary_file.exists():
                    final_summary = read_json(final_summary_file)
                    if final_summary["quantization_protocol_sha256"] != protocol_sha256:
                        raise RuntimeError(
                            "Tamamlanmış çalışma farklı quantization protokolüne ait."
                        )
                    metrics_document = read_json(
                        Path(final_summary["test_metrics_file"])
                    )
                    metrics = {
                        key: value
                        for key, value in metrics_document.items()
                        if key
                        not in {
                            "generated_at_utc",
                            "model_name",
                            "method_name",
                            "seed",
                            "task_name",
                            "test_evaluation_attempt_count",
                            "test_used_for_selection",
                        }
                    }
                    artifact_metadata = read_json(
                        Path(final_summary["artifact_metadata_file"])
                    )
                    training_summary = (
                        read_json(Path(final_summary["training_summary_file"]))
                        if final_summary.get("training_summary_file")
                        else None
                    )
                    print("Tamamlanmış sonuç yeniden kullanılıyor.")
                else:
                    if method_name in {"DQ", "PTQ"}:
                        metrics, artifact_metadata = run_onnx_method(
                            method_name=method_name,
                            model_name=model_name,
                            seed=seed,
                            fp32_onnx_file=fp32_onnx_file,
                            run_directory=run_directory,
                            assets=assets,
                            protocol=protocol,
                            calibration_batches=calibration_batches,
                            restart_incomplete=args.restart_incomplete,
                        )
                        training_summary = None
                    else:
                        training_summary = train_qat_model(
                            model_name=model_name,
                            seed=seed,
                            source_model_state_dict=source_state,
                            source_checkpoint_file=source_checkpoint_file,
                            assets=assets,
                            fp32_learning_rate=fp32_learning_rate,
                            protocol=protocol,
                            protocol_sha256=protocol_sha256,
                            backend_config=backend_config,
                            backend_config_source=backend_config_source,
                            run_directory=run_directory,
                        )
                        metrics, artifact_metadata = evaluate_qat_test_once(
                            model_name=model_name,
                            seed=seed,
                            source_model_state_dict=source_state,
                            assets=assets,
                            protocol=protocol,
                            backend_config=backend_config,
                            training_summary=training_summary,
                            run_directory=run_directory,
                            restart_incomplete=args.restart_incomplete,
                        )

                    final_summary = {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model_name": model_name,
                        "method_name": method_name,
                        "seed": int(seed),
                        "quantization_protocol_file": str(protocol_file),
                        "quantization_protocol_sha256": protocol_sha256,
                        "source_checkpoint_file": str(source_checkpoint_file),
                        "source_checkpoint_sha256": calculate_sha256(
                            source_checkpoint_file
                        ),
                        "test_metrics_file": str(run_directory / "test_metrics.json"),
                        "artifact_metadata_file": str(
                            run_directory / "artifact_metadata.json"
                        ),
                        "training_summary_file": (
                            str(run_directory / "training_summary.json")
                            if method_name == "QAT"
                            else None
                        ),
                        "test_evaluation_attempt_count": 1,
                        "test_used_for_selection": False,
                        "validation_passed": True,
                    }
                    write_json_atomic(final_summary, final_summary_file)

                test_macro_f1 = float(metrics["macro_f1"])
                fp32_macro_f1 = float(fp32_reference["test_macro_f1"])

                run_records.append(
                    {
                        "model_name": model_name,
                        "method_name": method_name,
                        "seed": int(seed),
                        "fp32_test_macro_f1": fp32_macro_f1,
                        "test_loss": float(metrics["loss"]),
                        "test_accuracy": float(metrics["accuracy"]),
                        "test_balanced_accuracy": float(
                            metrics["balanced_accuracy"]
                        ),
                        "test_macro_precision": float(
                            metrics["macro_precision"]
                        ),
                        "test_macro_recall": float(metrics["macro_recall"]),
                        "test_macro_f1": test_macro_f1,
                        "test_weighted_f1": float(metrics["weighted_f1"]),
                        "fp32_test_mcc": float(fp32_reference["test_mcc"]),
                        "test_mcc": float(
                            metrics["matthews_correlation_coefficient"]
                        ),
                        "macro_f1_delta_vs_fp32": float(
                            test_macro_f1 - fp32_macro_f1
                        ),
                        "test_sample_count": int(metrics["sample_count"]),
                        "test_elapsed_seconds": float(metrics["elapsed_seconds"]),
                        "test_throughput_samples_per_second": float(
                            metrics["throughput_samples_per_second"]
                        ),
                        "artifact_size_bytes": int(
                            artifact_metadata["artifact_size_bytes"]
                        ),
                        "quantized_node_or_module_count": int(
                            artifact_metadata.get(
                                "quantized_module_count",
                                0,
                            )
                            + artifact_metadata.get(
                                "quantized_graph_node_count",
                                artifact_metadata.get("graph_summary", {}).get(
                                    "quantized_node_count",
                                    0,
                                ),
                            )
                        ),
                        "qat_best_epoch": (
                            int(training_summary["best_epoch"])
                            if training_summary is not None
                            else None
                        ),
                        "qat_best_validation_macro_f1_converted_int8": (
                            float(
                                training_summary[
                                    "best_validation_macro_f1_converted_int8"
                                ]
                            )
                            if training_summary is not None
                            else None
                        ),
                        "test_evaluation_attempt_count": 1,
                        "test_used_for_selection": False,
                        "run_directory": str(run_directory),
                        "artifact_file": str(artifact_metadata["artifact_file"]),
                    }
                )

                for class_index, class_name in enumerate(TARGET_CLASSES):
                    class_metrics = metrics["per_class"][class_name]
                    per_class_records.append(
                        {
                            "model_name": model_name,
                            "method_name": method_name,
                            "seed": int(seed),
                            "class_index": int(class_index),
                            "class_name": class_name,
                            "support": int(class_metrics["support"]),
                            "predicted_count": int(
                                class_metrics["predicted_count"]
                            ),
                            "precision": float(class_metrics["precision"]),
                            "recall": float(class_metrics["recall"]),
                            "f1": float(class_metrics["f1"]),
                            "false_negative_rate": float(
                                class_metrics["false_negative_rate"]
                            ),
                        }
                    )

                matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
                for true_index, true_class in enumerate(TARGET_CLASSES):
                    for predicted_index, predicted_class in enumerate(
                        TARGET_CLASSES
                    ):
                        confusion_records.append(
                            {
                                "model_name": model_name,
                                "method_name": method_name,
                                "seed": int(seed),
                                "true_class_index": int(true_index),
                                "true_class": true_class,
                                "predicted_class_index": int(predicted_index),
                                "predicted_class": predicted_class,
                                "sample_count": int(
                                    matrix[true_index, predicted_index]
                                ),
                            }
                        )

                print(f"Test Macro F1       : {test_macro_f1:.6f}")
                print(
                    "FP32'ye göre değişim: "
                    f"{test_macro_f1 - fp32_macro_f1:+.6f}"
                )
                print("Test seçime katıldı: False")

    run_frame = pd.DataFrame(run_records)
    per_class_frame = pd.DataFrame(per_class_records)
    confusion_frame = pd.DataFrame(confusion_records)

    if len(run_frame) != 45:
        raise RuntimeError("Quantization sonuç sayısı 45 değil.")
    if run_frame.duplicated(
        subset=["model_name", "method_name", "seed"]
    ).any():
        raise RuntimeError("Tekrarlanan model-yöntem-seed sonucu var.")
    if (run_frame["test_evaluation_attempt_count"] != 1).any():
        raise RuntimeError("Test değerlendirme girişimlerinden biri 1 değil.")
    if run_frame["test_used_for_selection"].any():
        raise RuntimeError("Test sonuçlarından biri model seçimine katılmış.")
    if (run_frame["quantized_node_or_module_count"] <= 0).any():
        raise RuntimeError("Sonuçlardan birinde quantized kapsam bulunmuyor.")

    aggregate_frame = aggregate_results(run_frame)
    per_class_aggregate_frame = aggregate_per_class(per_class_frame)

    run_report_file = (
        report_directory / "nbaiot_family3_quantization_runs.csv"
    )
    aggregate_report_file = (
        report_directory / "nbaiot_family3_quantization_aggregate.csv"
    )
    per_class_run_file = (
        report_directory / "nbaiot_family3_quantization_per_class_runs.csv"
    )
    per_class_aggregate_file = (
        report_directory / "nbaiot_family3_quantization_per_class_aggregate.csv"
    )
    confusion_report_file = (
        report_directory / "nbaiot_family3_quantization_confusion_matrices.csv"
    )
    summary_file = (
        report_directory / "nbaiot_family3_quantization_summary.json"
    )

    write_csv_atomic(run_frame, run_report_file)
    write_csv_atomic(aggregate_frame, aggregate_report_file)
    write_csv_atomic(per_class_frame, per_class_run_file)
    write_csv_atomic(per_class_aggregate_frame, per_class_aggregate_file)
    write_csv_atomic(confusion_frame, confusion_report_file)

    ranking_frame = aggregate_frame[
        aggregate_frame["metric"] == "test_macro_f1"
    ].copy()
    ranking_frame = ranking_frame.sort_values(
        by=["mean", "standard_deviation"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking_records = [
        {
            "rank": int(index + 1),
            "model_name": str(row.model_name),
            "method_name": str(row.method_name),
            "mean_test_macro_f1": float(row.mean),
            "standard_deviation": float(row.standard_deviation),
            "ci95_lower": float(row.ci95_lower),
            "ci95_upper": float(row.ci95_upper),
        }
        for index, row in enumerate(ranking_frame.itertuples(index=False))
    ]

    write_json_atomic(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_name": "N-BaIoT family_3 DQ PTQ QAT multi-seed",
            "protocol_file": str(protocol_file),
            "protocol_sha256": protocol_sha256,
            "models": list(MODEL_NAMES),
            "methods": list(METHOD_NAMES),
            "seeds": list(SEEDS),
            "total_run_count": int(len(run_frame)),
            "primary_metric": "test_macro_f1",
            "security_metric": "per_class_false_negative_rate",
            "test_evaluations_per_model_method_seed": 1,
            "test_used_for_model_selection": False,
            "macro_f1_ranking": ranking_records,
            "run_report": str(run_report_file),
            "aggregate_report": str(aggregate_report_file),
            "per_class_run_report": str(per_class_run_file),
            "per_class_aggregate_report": str(per_class_aggregate_file),
            "confusion_matrix_report": str(confusion_report_file),
            "native_artifact_size_note": (
                "ONNX and PyTorch quantized state_dict sizes are runtime-specific. "
                "Cross-runtime size comparison is deferred to the standardized "
                "deployment audit stage."
            ),
            "statistical_hypothesis_testing_status": (
                "Not yet performed; paired FP32-versus-quantization tests are next."
            ),
            "validation_passed": True,
        },
        summary_file,
    )

    print()
    print("=" * 78)
    print("DQ, PTQ ve QAT Ana Deneyleri Tamamlandı")
    print("=" * 78)

    for record in ranking_records:
        print(
            f"{record['rank']:2d}. "
            f"{record['model_name']:14s} "
            f"{record['method_name']:3s} | "
            f"Macro F1={record['mean_test_macro_f1']:.6f} "
            f"± {record['standard_deviation']:.6f} | "
            f"%95 GA=[{record['ci95_lower']:.6f}, "
            f"{record['ci95_upper']:.6f}]"
        )

    print()
    print(f"Tamamlanan sonuç       : {len(run_frame)}")
    print("Test değerlendirmesi   : 45")
    print("Test seçime katıldı    : False")
    print("Quantized kapsam       : True")
    print("Doğrulama geçti        : True")
    print()
    print(f"Seed sonuçları         : {run_report_file}")
    print(f"Toplu istatistikler    : {aggregate_report_file}")
    print(f"Sınıf sonuçları        : {per_class_run_file}")
    print(f"Sınıf istatistikleri   : {per_class_aggregate_file}")
    print(f"Karışıklık matrisleri  : {confusion_report_file}")
    print(f"JSON özet              : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
