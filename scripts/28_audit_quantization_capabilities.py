"""
N-BaIoT TinyML IDS quantization uyumluluk ve dönüşüm denetimi.

Denetim kapsamı:
1. Python, PyTorch, ONNX ve ONNX Runtime sürümleri.
2. Yerel CPU ve quantized engine bilgileri.
3. torchao ve PyTorch quantization API sembollerinin varlığı.
4. Üç FP32 seed-42 modelinin ONNX dışa aktarımı.
5. PyTorch ile ONNX Runtime FP32 çıktı uyumu.
6. ONNX Runtime Dynamic Quantization dönüşümü.
7. Train-only kalibrasyonla ONNX Runtime statik PTQ dönüşümü.
8. Dönüştürülmüş grafiklerdeki quantized operatör ve tensor kapsamı.
9. FP32, DQ ve PTQ artifact dosya boyutları.
10. Quantize edilmiş modellerin FP32 ONNX çıktısına göre
    sınıflandırma uyumu.

Bilimsel sınırlar:
- Kalibrasyon yalnızca train splitinden alınır.
- Validation ve test splitleri kullanılmaz.
- Bu aşama nihai doğruluk veya gecikme deneyi değildir.
- DQ/PTQ dönüşümünün başarılı olması, modelin bütünüyle INT8
  çalıştığı anlamına gelmez. Gerçek grafik kapsamı ayrıca kaydedilir.
- QAT bu aşamada çalıştırılmaz; yerel QAT API uyumluluğu denetlenir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn


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
    / "audit_v1"
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

QUANTIZED_NODE_TYPES = {
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "QLinearConv",
    "ConvInteger",
    "QGemm",
}

INTEGER_TENSOR_TYPE_NAMES = {
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
}


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Yerel PyTorch/ONNX Runtime quantization "
            "uyumluluğunu denetler."
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
        "--comparison-batch-size",
        type=int,
        default=256,
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
        "--torch-threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ==========================================================
# GENERIC HELPERS
# ==========================================================

def json_default(
    value: object,
) -> object:
    """NumPy, Path ve set nesnelerini JSON uyumlu hâle getirir."""

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

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def read_json(
    input_file: Path,
) -> dict[str, Any]:
    """JSON dosyasını yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        document = json.load(file_handle)

    if not isinstance(document, dict):
        raise TypeError(
            f"JSON kökü sözlük değil: {input_file}"
        )

    return document


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

    temporary_file.replace(output_file)


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

    temporary_file.replace(output_file)


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

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def file_size_bytes(
    file_path: Path,
) -> int:
    """Dosya boyutunu byte olarak döndürür."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {file_path}"
        )

    return int(file_path.stat().st_size)


def safe_error_text(
    error: BaseException,
) -> str:
    """Hata metnini tek satırlık güvenli biçime getirir."""

    return " ".join(
        str(error).strip().split()
    )


def package_available(
    module_name: str,
) -> bool:
    """Bir Python modülünün bulunup bulunmadığını kontrol eder."""

    try:
        return (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )

    except (
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ):
        return False


def get_module_version(
    module_name: str,
) -> str | None:
    """Kurulu modülün sürümünü almaya çalışır."""

    if not package_available(module_name):
        return None

    try:
        module = importlib.import_module(
            module_name
        )

    except Exception:
        return None

    version = getattr(
        module,
        "__version__",
        None,
    )

    if version is None:
        return "unknown"

    return str(version)


def check_symbol(
    module_name: str,
    symbol_path: str,
) -> dict[str, Any]:
    """Bir modül içindeki noktalı sembol yolunun varlığını denetler."""

    result: dict[str, Any] = {
        "module_name": module_name,
        "symbol_path": symbol_path,
        "module_available": False,
        "symbol_available": False,
        "object_type": None,
        "signature": None,
        "error": None,
    }

    try:
        module = importlib.import_module(
            module_name
        )

        result["module_available"] = True

        current_object: Any = module

        for part in symbol_path.split("."):
            current_object = getattr(
                current_object,
                part,
            )

        result["symbol_available"] = True

        result["object_type"] = (
            type(current_object).__name__
        )

        if callable(current_object):
            try:
                result["signature"] = str(
                    inspect.signature(
                        current_object
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                result["signature"] = (
                    "signature_unavailable"
                )

    except Exception as error:
        result["error"] = safe_error_text(
            error
        )

    return result


# ==========================================================
# CHECKPOINT AND DATA HELPERS
# ==========================================================

def load_checkpoint(
    checkpoint_file: Path,
) -> dict[str, Any]:
    """PyTorch checkpointini CPU üzerinde yükler."""

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
    """FP32 checkpoint kimliğini doğrular."""

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


def obtain_train_batches(
    *,
    assets: Any,
    batch_size: int,
    batch_count: int,
    seed: int,
) -> list[np.ndarray]:
    """Train splitinden deterministik kalibrasyon batchleri alır."""

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
        if features.dtype != torch.float32:
            raise TypeError(
                "Kalibrasyon özellikleri float32 değil."
            )

        if not torch.isfinite(features).all():
            raise RuntimeError(
                "Kalibrasyon batchinde NaN veya Inf bulundu."
            )

        batch_array = np.ascontiguousarray(
            features.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        batches.append(batch_array)

        if len(batches) >= batch_count:
            break

    if len(batches) != batch_count:
        raise RuntimeError(
            "İstenen sayıda train kalibrasyon batchi alınamadı: "
            f"istenen={batch_count}, bulunan={len(batches)}"
        )

    return batches


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
    """PyTorch modelini ONNX biçimine dışa aktarır."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.cpu()
    model.eval()

    sample_features = (
        sample_features.detach()
        .cpu()
        .to(dtype=torch.float32)
    )

    export_arguments = {
        "model": model,
        "args": (sample_features,),
        "f": str(output_file),
        "input_names": ["features"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "features": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
        },
        "opset_version": int(
            opset_version
        ),
        "do_constant_folding": True,
        "export_params": True,
    }

    try:
        torch.onnx.export(
            **export_arguments,
            dynamo=False,
        )

    except TypeError:
        torch.onnx.export(
            **export_arguments
        )

    if not output_file.exists():
        raise RuntimeError(
            "ONNX dışa aktarımı dosya oluşturmadı."
        )

    if output_file.stat().st_size <= 0:
        raise RuntimeError(
            "ONNX dışa aktarımı boş dosya oluşturdu."
        )


def validate_onnx_model(
    model_file: Path,
) -> None:
    """ONNX modelini onnx.checker ile doğrular."""

    import onnx

    model = onnx.load(
        str(model_file)
    )

    onnx.checker.check_model(
        model
    )


def summarize_onnx_graph(
    model_file: Path,
) -> dict[str, Any]:
    """ONNX grafiğinin operatör ve tensor türlerini özetler."""

    import onnx

    model = onnx.load(
        str(model_file)
    )

    node_type_counts = Counter(
        node.op_type
        for node in model.graph.node
    )

    initializer_type_counts = Counter(
        int(initializer.data_type)
        for initializer in model.graph.initializer
    )

    integer_initializer_count = int(
        sum(
            count
            for data_type, count
            in initializer_type_counts.items()
            if data_type in INTEGER_TENSOR_TYPE_NAMES
        )
    )

    total_initializer_count = int(
        len(model.graph.initializer)
    )

    quantized_node_count = int(
        sum(
            node_type_counts.get(
                node_type,
                0,
            )
            for node_type in QUANTIZED_NODE_TYPES
        )
    )

    quantized_node_types_present = sorted(
        node_type
        for node_type in QUANTIZED_NODE_TYPES
        if node_type_counts.get(
            node_type,
            0,
        )
        > 0
    )

    initializer_type_names = {
        INTEGER_TENSOR_TYPE_NAMES.get(
            data_type,
            str(data_type),
        ): int(count)
        for data_type, count
        in sorted(
            initializer_type_counts.items()
        )
    }

    return {
        "node_count": int(
            len(model.graph.node)
        ),
        "node_type_counts": dict(
            sorted(
                node_type_counts.items()
            )
        ),
        "quantized_node_count": (
            quantized_node_count
        ),
        "quantized_node_types_present": (
            quantized_node_types_present
        ),
        "initializer_count": (
            total_initializer_count
        ),
        "integer_initializer_count": (
            integer_initializer_count
        ),
        "integer_initializer_ratio": float(
            integer_initializer_count
            / total_initializer_count
            if total_initializer_count > 0
            else 0.0
        ),
        "initializer_type_counts": (
            initializer_type_names
        ),
    }


def create_ort_session(
    *,
    model_file: Path,
    torch_threads: int,
) -> Any:
    """ONNX Runtime CPU oturumu oluşturur."""

    import onnxruntime as ort

    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = int(
        torch_threads
    )

    session_options.inter_op_num_threads = 1

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_file),
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider",
        ],
    )

    return session


def run_ort_model(
    *,
    model_file: Path,
    input_array: np.ndarray,
    torch_threads: int,
) -> np.ndarray:
    """ONNX Runtime modeli çalıştırır."""

    session = create_ort_session(
        model_file=model_file,
        torch_threads=torch_threads,
    )

    input_metadata = (
        session.get_inputs()[0]
    )

    output_values = session.run(
        None,
        {
            input_metadata.name: np.ascontiguousarray(
                input_array.astype(
                    np.float32,
                    copy=False,
                )
            ),
        },
    )

    if len(output_values) != 1:
        raise RuntimeError(
            "ONNX Runtime tek çıktı üretmedi."
        )

    output_array = np.asarray(
        output_values[0],
        dtype=np.float32,
    )

    if not np.isfinite(
        output_array
    ).all():
        raise RuntimeError(
            "ONNX Runtime çıktısında NaN veya Inf bulundu."
        )

    return output_array


def compare_logits(
    *,
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """İki logit matrisini karşılaştırır."""

    reference = np.asarray(
        reference_logits,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate_logits,
        dtype=np.float64,
    )

    if reference.shape != candidate.shape:
        raise RuntimeError(
            "Logit boyutları uyuşmuyor: "
            f"referans={reference.shape}, "
            f"aday={candidate.shape}"
        )

    differences = np.abs(
        candidate - reference
    )

    reference_predictions = np.argmax(
        reference,
        axis=1,
    )

    candidate_predictions = np.argmax(
        candidate,
        axis=1,
    )

    prediction_agreement = float(
        np.mean(
            reference_predictions
            == candidate_predictions
        )
    )

    return {
        "sample_count": int(
            reference.shape[0]
        ),
        "output_shape": list(
            reference.shape
        ),
        "maximum_absolute_difference": float(
            differences.max()
            if differences.size > 0
            else 0.0
        ),
        "mean_absolute_difference": float(
            differences.mean()
            if differences.size > 0
            else 0.0
        ),
        "root_mean_squared_difference": float(
            np.sqrt(
                np.mean(
                    np.square(
                        candidate - reference
                    )
                )
            )
            if differences.size > 0
            else 0.0
        ),
        "argmax_prediction_agreement": (
            prediction_agreement
        ),
        "allclose": bool(
            np.allclose(
                candidate,
                reference,
                rtol=rtol,
                atol=atol,
            )
        ),
        "rtol": float(rtol),
        "atol": float(atol),
    }


# ==========================================================
# QUANTIZATION HELPERS
# ==========================================================

def create_calibration_reader(
    *,
    calibration_reader_base: type,
    input_name: str,
    batches: list[np.ndarray],
) -> Any:
    """ONNX Runtime CalibrationDataReader nesnesi oluşturur."""

    class TrainOnlyCalibrationReader(
        calibration_reader_base
    ):
        """Bellekteki train batchlerini kalibrasyon için döndürür."""

        def __init__(
            self,
            model_input_name: str,
            calibration_batches: list[np.ndarray],
        ) -> None:
            self.model_input_name = (
                model_input_name
            )

            self.calibration_batches = [
                np.ascontiguousarray(
                    batch.astype(
                        np.float32,
                        copy=False,
                    )
                )
                for batch in calibration_batches
            ]

            self.rewind()

        def get_next(
            self,
        ) -> dict[str, np.ndarray] | None:
            try:
                batch = next(
                    self._iterator
                )

            except StopIteration:
                return None

            return {
                self.model_input_name: batch,
            }

        def rewind(
            self,
        ) -> None:
            self._iterator = iter(
                self.calibration_batches
            )

    return TrainOnlyCalibrationReader(
        model_input_name=input_name,
        calibration_batches=batches,
    )


def call_dynamic_quantization(
    *,
    input_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    """ONNX Runtime dynamic quantization dönüşümünü çalıştırır."""

    from onnxruntime.quantization import (
        QuantType,
        quantize_dynamic,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    call_arguments: dict[str, Any] = {
        "model_input": str(
            input_file
        ),
        "model_output": str(
            output_file
        ),
        "weight_type": QuantType.QInt8,
        "per_channel": True,
        "reduce_range": False,
    }

    function_signature = inspect.signature(
        quantize_dynamic
    )

    accepted_arguments = {
        key: value
        for key, value in call_arguments.items()
        if key in function_signature.parameters
    }

    quantize_dynamic(
        **accepted_arguments
    )

    if not output_file.exists():
        raise RuntimeError(
            "Dynamic quantization dosya oluşturmadı."
        )

    return {
        "function_signature": str(
            function_signature
        ),
        "used_arguments": {
            key: str(value)
            for key, value
            in accepted_arguments.items()
            if key not in {
                "model_input",
                "model_output",
            }
        },
    }


def call_static_quantization(
    *,
    input_file: Path,
    output_file: Path,
    calibration_batches: list[np.ndarray],
    input_name: str,
) -> dict[str, Any]:
    """Train-only kalibrasyonla ONNX Runtime PTQ çalıştırır."""

    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_reader = (
        create_calibration_reader(
            calibration_reader_base=(
                CalibrationDataReader
            ),
            input_name=input_name,
            batches=calibration_batches,
        )
    )

    call_arguments: dict[str, Any] = {
        "model_input": str(
            input_file
        ),
        "model_output": str(
            output_file
        ),
        "calibration_data_reader": (
            calibration_reader
        ),
        "quant_format": QuantFormat.QDQ,
        "activation_type": QuantType.QUInt8,
        "weight_type": QuantType.QInt8,
        "per_channel": True,
        "reduce_range": False,
        "calibrate_method": (
            CalibrationMethod.MinMax
        ),
    }

    function_signature = inspect.signature(
        quantize_static
    )

    accepted_arguments = {
        key: value
        for key, value in call_arguments.items()
        if key in function_signature.parameters
    }

    quantize_static(
        **accepted_arguments
    )

    if not output_file.exists():
        raise RuntimeError(
            "Statik PTQ dosya oluşturmadı."
        )

    return {
        "function_signature": str(
            function_signature
        ),
        "calibration_split": "train",
        "calibration_batch_count": int(
            len(calibration_batches)
        ),
        "calibration_sample_count": int(
            sum(
                len(batch)
                for batch in calibration_batches
            )
        ),
        "used_arguments": {
            key: str(value)
            for key, value
            in accepted_arguments.items()
            if key not in {
                "model_input",
                "model_output",
                "calibration_data_reader",
            }
        },
    }


def run_quantization_conversion(
    *,
    conversion_name: str,
    conversion_function: Callable[[], dict[str, Any]],
    output_file: Path,
    reference_logits: np.ndarray,
    comparison_features: np.ndarray,
    torch_threads: int,
) -> dict[str, Any]:
    """Bir quantization dönüşümünü çalıştırıp sonucunu denetler."""

    result: dict[str, Any] = {
        "conversion_name": conversion_name,
        "conversion_attempted": True,
        "conversion_succeeded": False,
        "validation_succeeded": False,
        "output_file": str(
            output_file
        ),
        "file_size_bytes": None,
        "sha256": None,
        "graph_summary": None,
        "output_comparison": None,
        "conversion_details": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }

    try:
        conversion_details = (
            conversion_function()
        )

        validate_onnx_model(
            output_file
        )

        candidate_logits = run_ort_model(
            model_file=output_file,
            input_array=comparison_features,
            torch_threads=torch_threads,
        )

        graph_summary = summarize_onnx_graph(
            output_file
        )

        output_comparison = compare_logits(
            reference_logits=reference_logits,
            candidate_logits=candidate_logits,
            rtol=0.0,
            atol=0.0,
        )

        result.update(
            {
                "conversion_succeeded": True,
                "validation_succeeded": True,
                "file_size_bytes": (
                    file_size_bytes(
                        output_file
                    )
                ),
                "sha256": calculate_sha256(
                    output_file
                ),
                "graph_summary": (
                    graph_summary
                ),
                "output_comparison": (
                    output_comparison
                ),
                "conversion_details": (
                    conversion_details
                ),
            }
        )

    except Exception as error:
        result[
            "error_type"
        ] = type(error).__name__

        result[
            "error_message"
        ] = safe_error_text(
            error
        )

        result[
            "traceback"
        ] = traceback.format_exc()

    return result


# ==========================================================
# ENVIRONMENT AUDIT
# ==========================================================

def build_environment_audit() -> dict[str, Any]:
    """Yerel yazılım ve quantization API ortamını denetler."""

    supported_engines: list[str] = []

    active_engine: str | None = None

    try:
        supported_engines = list(
            torch.backends.quantized.supported_engines
        )

    except Exception:
        supported_engines = []

    try:
        active_engine = str(
            torch.backends.quantized.engine
        )

    except Exception:
        active_engine = None

    symbol_checks = [
        check_symbol(
            "torch",
            "ao.quantization.quantize_dynamic",
        ),
        check_symbol(
            "torch.ao.quantization.quantize_fx",
            "prepare_fx",
        ),
        check_symbol(
            "torch.ao.quantization.quantize_fx",
            "prepare_qat_fx",
        ),
        check_symbol(
            "torch.ao.quantization.quantize_fx",
            "convert_fx",
        ),
        check_symbol(
            "torch.ao.quantization",
            "get_default_qconfig_mapping",
        ),
        check_symbol(
            "torch.ao.quantization",
            "get_default_qat_qconfig_mapping",
        ),
        check_symbol(
            "torchao.quantization",
            "quantize_",
        ),
        check_symbol(
            "torchao.quantization.pt2e",
            "prepare_pt2e",
        ),
        check_symbol(
            "torchao.quantization.pt2e",
            "prepare_qat_pt2e",
        ),
        check_symbol(
            "torchao.quantization.pt2e",
            "convert_pt2e",
        ),
        check_symbol(
            "onnxruntime.quantization",
            "quantize_dynamic",
        ),
        check_symbol(
            "onnxruntime.quantization",
            "quantize_static",
        ),
    ]

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "operating_system": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_architecture": (
            platform.architecture()[0]
        ),
        "torch_version": str(
            torch.__version__
        ),
        "torch_cuda_available": bool(
            torch.cuda.is_available()
        ),
        "torch_cuda_version": (
            torch.version.cuda
        ),
        "torch_num_threads": int(
            torch.get_num_threads()
        ),
        "quantized_supported_engines": (
            supported_engines
        ),
        "quantized_active_engine": (
            active_engine
        ),
        "packages": {
            "numpy": get_module_version(
                "numpy"
            ),
            "pandas": get_module_version(
                "pandas"
            ),
            "torch": get_module_version(
                "torch"
            ),
            "torchao": get_module_version(
                "torchao"
            ),
            "onnx": get_module_version(
                "onnx"
            ),
            "onnxruntime": get_module_version(
                "onnxruntime"
            ),
        },
        "package_availability": {
            "torchao": package_available(
                "torchao"
            ),
            "onnx": package_available(
                "onnx"
            ),
            "onnxruntime": package_available(
                "onnxruntime"
            ),
            "onnxruntime_quantization": (
                package_available(
                    "onnxruntime.quantization"
                )
            ),
        },
        "api_symbol_checks": symbol_checks,
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Quantization yetenek ve dönüşüm denetimini çalıştırır."""

    args = parse_arguments()

    if args.comparison_batch_size <= 0:
        raise ValueError(
            "comparison-batch-size pozitif olmalıdır."
        )

    if args.calibration_batch_size <= 0:
        raise ValueError(
            "calibration-batch-size pozitif olmalıdır."
        )

    if args.calibration_batches <= 0:
        raise ValueError(
            "calibration-batches pozitif olmalıdır."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads pozitif olmalıdır."
        )

    if args.opset_version <= 0:
        raise ValueError(
            "opset-version pozitif olmalıdır."
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

    audit_csv_file = (
        report_directory
        / "nbaiot_quantization_capability_audit.csv"
    )

    summary_json_file = (
        report_directory
        / "nbaiot_quantization_capability_audit_summary.json"
    )

    environment_json_file = (
        report_directory
        / "nbaiot_quantization_environment.json"
    )

    output_files = (
        audit_csv_file,
        summary_json_file,
        environment_json_file,
    )

    if (
        any(
            output_file.exists()
            for output_file in output_files
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "Quantization denetim çıktıları mevcut. "
            "--overwrite kullan."
        )

    environment_audit = (
        build_environment_audit()
    )

    write_json_atomic(
        document=environment_audit,
        output_file=environment_json_file,
    )

    required_packages = {
        "onnx": environment_audit[
            "package_availability"
        ]["onnx"],
        "onnxruntime": environment_audit[
            "package_availability"
        ]["onnxruntime"],
        "onnxruntime_quantization": environment_audit[
            "package_availability"
        ]["onnxruntime_quantization"],
    }

    missing_required_packages = [
        package_name
        for package_name, available
        in required_packages.items()
        if not available
    ]

    if missing_required_packages:
        raise ModuleNotFoundError(
            "Quantization denetimi için eksik paketler var: "
            + ", ".join(
                missing_required_packages
            )
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

    comparison_batches = obtain_train_batches(
        assets=assets,
        batch_size=args.comparison_batch_size,
        batch_count=1,
        seed=args.seed,
    )

    comparison_features_numpy = (
        comparison_batches[0]
    )

    comparison_features_torch = (
        torch.from_numpy(
            comparison_features_numpy
        )
    )

    calibration_batches = (
        obtain_train_batches(
            assets=assets,
            batch_size=(
                args.calibration_batch_size
            ),
            batch_count=(
                args.calibration_batches
            ),
            seed=args.seed,
        )
    )

    calibration_sample_count = int(
        sum(
            len(batch)
            for batch in calibration_batches
        )
    )

    print("=" * 78)
    print("N-BaIoT Quantization Uyumluluk ve Dönüşüm Denetimi")
    print("=" * 78)
    print(
        f"PyTorch           : "
        f"{environment_audit['torch_version']}"
    )
    print(
        f"ONNX              : "
        f"{environment_audit['packages']['onnx']}"
    )
    print(
        f"ONNX Runtime      : "
        f"{environment_audit['packages']['onnxruntime']}"
    )
    print(
        f"torchao           : "
        f"{environment_audit['packages']['torchao']}"
    )
    print(
        "Quantized engines : "
        + ", ".join(
            environment_audit[
                "quantized_supported_engines"
            ]
        )
    )
    print(
        "Aktif engine      : "
        f"{environment_audit['quantized_active_engine']}"
    )
    print(
        f"Kaynak seed       : {args.seed}"
    )
    print(
        f"Karşılaştırma     : "
        f"{len(comparison_features_numpy):,} train örneği"
    )
    print(
        f"PTQ kalibrasyonu  : "
        f"{calibration_sample_count:,} train örneği"
    )
    print(
        "Validation        : Kullanılmayacak"
    )
    print(
        "Test              : Kullanılmayacak"
    )
    print("=" * 78)

    model_records: list[
        dict[str, Any]
    ] = []

    model_details: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_name in MODEL_NAMES:
        print()
        print(
            f"{model_name}:"
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

        fp32_onnx_file = (
            model_directory
            / "model_fp32.onnx"
        )

        dynamic_onnx_file = (
            model_directory
            / "model_dynamic_int8.onnx"
        )

        static_onnx_file = (
            model_directory
            / "model_static_ptq_int8.onnx"
        )

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

        model = create_model(
            model_name=model_name,
            input_features=(
                assets.feature_count
            ),
            num_classes=(
                assets.class_count
            ),
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        model.cpu()
        model.eval()

        with torch.inference_mode():
            pytorch_logits = (
                model(
                    comparison_features_torch
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        if not np.isfinite(
            pytorch_logits
        ).all():
            raise RuntimeError(
                f"{model_name}: PyTorch logitlerinde "
                "NaN veya Inf bulundu."
            )

        export_model_to_onnx(
            model=model,
            sample_features=(
                comparison_features_torch[:8]
            ),
            output_file=fp32_onnx_file,
            opset_version=(
                args.opset_version
            ),
        )

        validate_onnx_model(
            fp32_onnx_file
        )

        fp32_ort_logits = run_ort_model(
            model_file=fp32_onnx_file,
            input_array=(
                comparison_features_numpy
            ),
            torch_threads=(
                args.torch_threads
            ),
        )

        fp32_parity = compare_logits(
            reference_logits=pytorch_logits,
            candidate_logits=fp32_ort_logits,
            rtol=1e-4,
            atol=1e-5,
        )

        if not fp32_parity["allclose"]:
            raise RuntimeError(
                f"{model_name}: PyTorch ile FP32 ONNX "
                "çıktıları tolerans içinde değil."
            )

        fp32_graph_summary = (
            summarize_onnx_graph(
                fp32_onnx_file
            )
        )

        import onnxruntime as ort

        fp32_session = create_ort_session(
            model_file=fp32_onnx_file,
            torch_threads=(
                args.torch_threads
            ),
        )

        input_name = (
            fp32_session.get_inputs()[0].name
        )

        dynamic_result = (
            run_quantization_conversion(
                conversion_name=(
                    "onnxruntime_dynamic_int8"
                ),
                conversion_function=lambda: (
                    call_dynamic_quantization(
                        input_file=(
                            fp32_onnx_file
                        ),
                        output_file=(
                            dynamic_onnx_file
                        ),
                    )
                ),
                output_file=dynamic_onnx_file,
                reference_logits=(
                    fp32_ort_logits
                ),
                comparison_features=(
                    comparison_features_numpy
                ),
                torch_threads=(
                    args.torch_threads
                ),
            )
        )

        static_result = (
            run_quantization_conversion(
                conversion_name=(
                    "onnxruntime_static_ptq_int8"
                ),
                conversion_function=lambda: (
                    call_static_quantization(
                        input_file=(
                            fp32_onnx_file
                        ),
                        output_file=(
                            static_onnx_file
                        ),
                        calibration_batches=(
                            calibration_batches
                        ),
                        input_name=input_name,
                    )
                ),
                output_file=static_onnx_file,
                reference_logits=(
                    fp32_ort_logits
                ),
                comparison_features=(
                    comparison_features_numpy
                ),
                torch_threads=(
                    args.torch_threads
                ),
            )
        )

        fp32_size = file_size_bytes(
            fp32_onnx_file
        )

        dynamic_size = (
            dynamic_result[
                "file_size_bytes"
            ]
        )

        static_size = (
            static_result[
                "file_size_bytes"
            ]
        )

        dynamic_graph = (
            dynamic_result.get(
                "graph_summary"
            )
            or {}
        )

        static_graph = (
            static_result.get(
                "graph_summary"
            )
            or {}
        )

        dynamic_comparison = (
            dynamic_result.get(
                "output_comparison"
            )
            or {}
        )

        static_comparison = (
            static_result.get(
                "output_comparison"
            )
            or {}
        )

        model_record = {
            "model_name": model_name,
            "source_seed": int(
                args.seed
            ),
            "checkpoint_file": str(
                checkpoint_file
            ),
            "checkpoint_sha256": (
                calculate_sha256(
                    checkpoint_file
                )
            ),
            "opset_version": int(
                args.opset_version
            ),
            "fp32_onnx_export_succeeded": True,
            "fp32_onnx_file_size_bytes": int(
                fp32_size
            ),
            "fp32_pytorch_ort_allclose": bool(
                fp32_parity[
                    "allclose"
                ]
            ),
            "fp32_pytorch_ort_max_abs_diff": float(
                fp32_parity[
                    "maximum_absolute_difference"
                ]
            ),
            "fp32_pytorch_ort_argmax_agreement": float(
                fp32_parity[
                    "argmax_prediction_agreement"
                ]
            ),
            "dynamic_quantization_succeeded": bool(
                dynamic_result[
                    "conversion_succeeded"
                ]
            ),
            "dynamic_validation_succeeded": bool(
                dynamic_result[
                    "validation_succeeded"
                ]
            ),
            "dynamic_file_size_bytes": (
                int(dynamic_size)
                if dynamic_size is not None
                else None
            ),
            "dynamic_size_ratio_vs_fp32": (
                float(
                    dynamic_size
                    / fp32_size
                )
                if dynamic_size is not None
                else None
            ),
            "dynamic_quantized_node_count": int(
                dynamic_graph.get(
                    "quantized_node_count",
                    0,
                )
            ),
            "dynamic_integer_initializer_count": int(
                dynamic_graph.get(
                    "integer_initializer_count",
                    0,
                )
            ),
            "dynamic_argmax_agreement_vs_fp32": (
                float(
                    dynamic_comparison.get(
                        "argmax_prediction_agreement"
                    )
                )
                if dynamic_comparison
                else None
            ),
            "dynamic_max_abs_diff_vs_fp32": (
                float(
                    dynamic_comparison.get(
                        "maximum_absolute_difference"
                    )
                )
                if dynamic_comparison
                else None
            ),
            "dynamic_error": (
                dynamic_result[
                    "error_message"
                ]
            ),
            "static_ptq_succeeded": bool(
                static_result[
                    "conversion_succeeded"
                ]
            ),
            "static_ptq_validation_succeeded": bool(
                static_result[
                    "validation_succeeded"
                ]
            ),
            "static_ptq_file_size_bytes": (
                int(static_size)
                if static_size is not None
                else None
            ),
            "static_ptq_size_ratio_vs_fp32": (
                float(
                    static_size
                    / fp32_size
                )
                if static_size is not None
                else None
            ),
            "static_ptq_quantized_node_count": int(
                static_graph.get(
                    "quantized_node_count",
                    0,
                )
            ),
            "static_ptq_integer_initializer_count": int(
                static_graph.get(
                    "integer_initializer_count",
                    0,
                )
            ),
            "static_ptq_argmax_agreement_vs_fp32": (
                float(
                    static_comparison.get(
                        "argmax_prediction_agreement"
                    )
                )
                if static_comparison
                else None
            ),
            "static_ptq_max_abs_diff_vs_fp32": (
                float(
                    static_comparison.get(
                        "maximum_absolute_difference"
                    )
                )
                if static_comparison
                else None
            ),
            "static_ptq_error": (
                static_result[
                    "error_message"
                ]
            ),
            "calibration_split": "train",
            "calibration_batch_count": int(
                args.calibration_batches
            ),
            "calibration_sample_count": int(
                calibration_sample_count
            ),
            "validation_split_used": False,
            "test_split_used": False,
        }

        model_records.append(
            model_record
        )

        model_details[
            model_name
        ] = {
            "checkpoint": {
                "path": str(
                    checkpoint_file
                ),
                "sha256": calculate_sha256(
                    checkpoint_file
                ),
            },
            "fp32_onnx": {
                "path": str(
                    fp32_onnx_file
                ),
                "sha256": calculate_sha256(
                    fp32_onnx_file
                ),
                "file_size_bytes": (
                    fp32_size
                ),
                "graph_summary": (
                    fp32_graph_summary
                ),
                "pytorch_ort_parity": (
                    fp32_parity
                ),
            },
            "dynamic_quantization": (
                dynamic_result
            ),
            "static_ptq": (
                static_result
            ),
        }

        print(
            "  FP32 ONNX | "
            f"boyut={fp32_size:,} byte | "
            f"PyTorch uyumu="
            f"{fp32_parity['allclose']} | "
            f"argmax="
            f"{fp32_parity['argmax_prediction_agreement']:.4f}"
        )

        if dynamic_result[
            "conversion_succeeded"
        ]:
            print(
                "  DQ        | "
                f"başarılı=True | "
                f"boyut={dynamic_size:,} byte | "
                f"q-düğüm="
                f"{dynamic_graph.get('quantized_node_count', 0)} | "
                f"argmax="
                f"{dynamic_comparison.get('argmax_prediction_agreement', 0.0):.4f}"
            )

        else:
            print(
                "  DQ        | "
                "başarılı=False | "
                f"hata={dynamic_result['error_message']}"
            )

        if static_result[
            "conversion_succeeded"
        ]:
            print(
                "  PTQ       | "
                f"başarılı=True | "
                f"boyut={static_size:,} byte | "
                f"q-düğüm="
                f"{static_graph.get('quantized_node_count', 0)} | "
                f"argmax="
                f"{static_comparison.get('argmax_prediction_agreement', 0.0):.4f}"
            )

        else:
            print(
                "  PTQ       | "
                "başarılı=False | "
                f"hata={static_result['error_message']}"
            )

    audit_frame = pd.DataFrame(
        model_records
    )

    if len(audit_frame) != len(
        MODEL_NAMES
    ):
        raise RuntimeError(
            "Quantization denetim model sayısı 3 değil."
        )

    all_fp32_exports_succeeded = bool(
        audit_frame[
            "fp32_onnx_export_succeeded"
        ].all()
    )

    all_fp32_parity_passed = bool(
        audit_frame[
            "fp32_pytorch_ort_allclose"
        ].all()
    )

    dynamic_supported_model_count = int(
        audit_frame[
            "dynamic_quantization_succeeded"
        ].sum()
    )

    static_ptq_supported_model_count = int(
        audit_frame[
            "static_ptq_succeeded"
        ].sum()
    )

    dynamic_models_with_quantized_nodes = int(
        (
            audit_frame[
                "dynamic_quantized_node_count"
            ]
            > 0
        ).sum()
    )

    static_models_with_quantized_nodes = int(
        (
            audit_frame[
                "static_ptq_quantized_node_count"
            ]
            > 0
        ).sum()
    )

    audit_completed = bool(
        all_fp32_exports_succeeded
        and all_fp32_parity_passed
    )

    if not audit_completed:
        raise RuntimeError(
            "Temel FP32 ONNX doğrulaması başarısız."
        )

    write_csv_atomic(
        frame=audit_frame,
        output_file=audit_csv_file,
    )

    qat_symbol_results = [
        result
        for result in environment_audit[
            "api_symbol_checks"
        ]
        if (
            "qat" in result[
                "symbol_path"
            ].lower()
            or "pt2e" in result[
                "symbol_path"
            ].lower()
            or (
                result["module_name"]
                == "torchao.quantization"
                and result["symbol_path"]
                == "quantize_"
            )
        )
    ]

    summary_document = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT local quantization capability "
            "and conversion audit"
        ),
        "analysis_version": "1.0",
        "source_seed": int(
            args.seed
        ),
        "models": list(
            MODEL_NAMES
        ),
        "feature_count": int(
            assets.feature_count
        ),
        "class_names": list(
            assets.target_classes
        ),
        "comparison_split": "train",
        "comparison_sample_count": int(
            len(
                comparison_features_numpy
            )
        ),
        "calibration_split": "train",
        "calibration_batch_size": int(
            args.calibration_batch_size
        ),
        "calibration_batch_count": int(
            args.calibration_batches
        ),
        "calibration_sample_count": int(
            calibration_sample_count
        ),
        "validation_split_used": False,
        "test_split_used": False,
        "all_fp32_onnx_exports_succeeded": (
            all_fp32_exports_succeeded
        ),
        "all_fp32_pytorch_ort_parity_passed": (
            all_fp32_parity_passed
        ),
        "dynamic_supported_model_count": (
            dynamic_supported_model_count
        ),
        "dynamic_models_with_quantized_nodes": (
            dynamic_models_with_quantized_nodes
        ),
        "static_ptq_supported_model_count": (
            static_ptq_supported_model_count
        ),
        "static_ptq_models_with_quantized_nodes": (
            static_models_with_quantized_nodes
        ),
        "torchao_installed": bool(
            environment_audit[
                "package_availability"
            ][
                "torchao"
            ]
        ),
        "qat_api_symbol_results": (
            qat_symbol_results
        ),
        "model_details": (
            model_details
        ),
        "interpretation_policy": {
            "conversion_success": (
                "The artifact was generated, passed ONNX validation "
                "and executed with ONNX Runtime."
            ),
            "quantized_coverage": (
                "Conversion success alone does not establish full INT8 "
                "coverage. Quantized graph nodes and integer initializers "
                "must also be inspected."
            ),
            "file_size": (
                "Tiny models may have quantization metadata overhead. "
                "A quantized ONNX file is not guaranteed to be smaller "
                "than the FP32 ONNX file."
            ),
            "accuracy": (
                "Argmax agreement in this audit uses train samples only "
                "and is not a final test performance result."
            ),
        },
        "environment_report": str(
            environment_json_file
        ),
        "audit_csv_report": str(
            audit_csv_file
        ),
        "artifact_directory": str(
            artifact_directory
        ),
        "audit_completed": (
            audit_completed
        ),
    }

    write_json_atomic(
        document=summary_document,
        output_file=summary_json_file,
    )

    print()
    print("=" * 78)
    print("Quantization Uyumluluk Denetimi Tamamlandı")
    print("=" * 78)
    print(
        "FP32 ONNX dışa aktarımı     : "
        f"{all_fp32_exports_succeeded}"
    )
    print(
        "PyTorch–ORT FP32 uyumu      : "
        f"{all_fp32_parity_passed}"
    )
    print(
        "DQ dönüşümü başarılı model  : "
        f"{dynamic_supported_model_count}/3"
    )
    print(
        "DQ quantized düğümlü model  : "
        f"{dynamic_models_with_quantized_nodes}/3"
    )
    print(
        "PTQ dönüşümü başarılı model : "
        f"{static_ptq_supported_model_count}/3"
    )
    print(
        "PTQ quantized düğümlü model : "
        f"{static_models_with_quantized_nodes}/3"
    )
    print(
        "torchao kurulu              : "
        f"{summary_document['torchao_installed']}"
    )
    print(
        "Validation kullanıldı       : False"
    )
    print(
        "Test kullanıldı             : False"
    )
    print(
        "Denetim tamamlandı          : "
        f"{audit_completed}"
    )
    print()
    print(
        f"CSV raporu      : {audit_csv_file}"
    )
    print(
        f"Ortam raporu    : {environment_json_file}"
    )
    print(
        f"JSON özet       : {summary_json_file}"
    )
    print(
        f"ONNX artifactları: {artifact_directory}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()