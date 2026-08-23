from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


ENGINE_VERSION = "phase5_qat_engine_v3_2"


class QATModelWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
    ) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = base_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        quantized = self.quant(inputs)
        outputs = self.model(quantized)
        return self.dequant(outputs)


def get_prepare_qat():
    candidate = getattr(
        torch.ao.quantization,
        "prepare_qat",
        None,
    )

    if candidate is not None:
        return candidate

    candidate = getattr(
        torch.quantization,
        "prepare_qat",
        None,
    )

    if candidate is not None:
        return candidate

    raise RuntimeError(
        "QAT prepare API is unavailable."
    )


def get_convert():
    candidate = getattr(
        torch.ao.quantization,
        "convert",
        None,
    )

    if candidate is not None:
        return candidate

    candidate = getattr(
        torch.quantization,
        "convert",
        None,
    )

    if candidate is not None:
        return candidate

    raise RuntimeError(
        "Quantization convert API is unavailable."
    )


def get_default_qat_qconfig(
    backend: str,
):
    candidate = getattr(
        torch.ao.quantization,
        "get_default_qat_qconfig",
        None,
    )

    if candidate is not None:
        return candidate(backend)

    candidate = getattr(
        torch.quantization,
        "get_default_qat_qconfig",
        None,
    )

    if candidate is not None:
        return candidate(backend)

    raise RuntimeError(
        "Default QAT qconfig API is unavailable."
    )


def prepare_qat_model(
    source_model: nn.Module,
    backend: str,
) -> nn.Module:
    wrapped = QATModelWrapper(
        copy.deepcopy(source_model)
    )

    wrapped.train()
    wrapped.qconfig = get_default_qat_qconfig(
        backend
    )

    prepared = get_prepare_qat()(
        wrapped,
        inplace=False,
    )

    prepared.train()
    return prepared


def convert_qat_model(
    prepared_model: nn.Module,
) -> nn.Module:
    conversion_source = copy.deepcopy(
        prepared_model
    )

    conversion_source.eval()

    converted = get_convert()(
        conversion_source,
        inplace=False,
    )

    converted.eval()
    return converted


def extract_logits(
    output: Any,
) -> torch.Tensor:
    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):
        for value in output:
            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    if isinstance(
        output,
        dict,
    ):
        for key in (
            "logits",
            "output",
            "outputs",
            "predictions",
        ):
            value = output.get(key)

            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    raise RuntimeError(
        "Model output did not contain logits."
    )


def float_linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(module.in_features),
            int(module.out_features),
        ]
        for module in model.modules()
        if isinstance(
            module,
            nn.Linear,
        )
    ]


def float_linear_count(
    model: nn.Module,
) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            nn.Linear,
        )
    )


def qat_linear_count(
    model: nn.Module,
) -> int:
    count = 0

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            ".qat." in module_path
            and class_name == "linear"
        ):
            count += 1

    return count


def static_quantized_linear_modules(
    model: nn.Module,
) -> list[nn.Module]:
    modules = []

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            "quantized" in module_path
            and ".qat." not in module_path
            and "dynamic" not in module_path
            and class_name == "linear"
        ):
            modules.append(module)

    return modules


def fake_quantizer_count(
    model: nn.Module,
) -> int:
    count = 0

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            "fake_quantize" in module_path
            or "fakequantize" in class_name
        ):
            count += 1

    return count


def quantized_weight_dtypes(
    model: nn.Module,
) -> list[str]:
    dtypes = []

    for module in static_quantized_linear_modules(
        model
    ):
        weight_getter = getattr(
            module,
            "weight",
            None,
        )

        if not callable(
            weight_getter
        ):
            raise RuntimeError(
                "Quantized Linear weight getter "
                "is unavailable."
            )

        dtypes.append(
            str(
                weight_getter().dtype
            )
        )

    return dtypes
