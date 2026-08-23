"""
TinyML IDS için fiziksel yapılandırılmış budama motoru.

Desteklenen mimariler:
- tinyml_mlp
- compact_dnn
- tiny_1d_cnn

Yöntem:
- Linear katmanların gizli nöronları fiziksel olarak kaldırılır.
- Conv1d katmanların çıktı kanalları fiziksel olarak kaldırılır.
- Çıkış sınıf birimleri budanmaz.
- Önem puanı, normalize edilmiş gelen ve giden L1 ağırlık
  büyüklüklerinin toplamıdır.
- weight_mask veya weight_orig tabanlı maskeli budama kullanılmaz.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.models.nbaiot_models import create_model


SUPPORTED_MODELS = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

SUPPORTED_PRUNING_RATIOS = (
    0.25,
    0.50,
    0.75,
)


@dataclass(frozen=True)
class StructuredPruningMetadata:
    """Fiziksel budama işleminin yeniden üretilebilir kaydı."""

    model_name: str
    requested_pruning_ratio: float
    pruning_name: str

    input_features: int
    num_classes: int

    original_architecture: dict[str, Any]
    compact_architecture: dict[str, Any]
    selected_indices: dict[str, list[int]]

    original_parameter_count: int
    compact_parameter_count: int
    removed_parameter_count: int
    parameter_reduction_ratio: float

    original_weight_layer_macs: int
    compact_weight_layer_macs: int
    removed_weight_layer_macs: int
    mac_reduction_ratio: float

    importance_method: str
    physical_compaction: bool
    mask_based_pruning: bool
    output_class_units_pruned: bool

    def to_dict(self) -> dict[str, Any]:
        """Metadata nesnesini JSON uyumlu sözlüğe dönüştürür."""

        return asdict(self)


def validate_pruning_ratio(
    pruning_ratio: float,
) -> float:
    """Budama oranını doğrular."""

    ratio = float(pruning_ratio)

    if not 0.0 < ratio < 1.0:
        raise ValueError(
            "Budama oranı 0 ile 1 arasında olmalıdır."
        )

    return ratio


def calculate_keep_count(
    original_count: int,
    pruning_ratio: float,
) -> int:
    """Budama sonrasında korunacak birim sayısını hesaplar."""

    if original_count <= 0:
        raise ValueError(
            "Özgün birim sayısı pozitif olmalıdır."
        )

    ratio = validate_pruning_ratio(
        pruning_ratio
    )

    # Yarım-yukarı yuvarlama.
    keep_count = int(
        math.floor(
            original_count
            * (1.0 - ratio)
            + 0.5
        )
    )

    return max(
        1,
        min(
            original_count,
            keep_count,
        ),
    )


def count_model_parameters(
    model: nn.Module,
) -> int:
    """Modeldeki toplam parametre sayısını döndürür."""

    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def calculate_parameter_memory_bytes(
    model: nn.Module,
) -> int:
    """Parametre tensorlerinin toplam belleğini byte olarak hesaplar."""

    return int(
        sum(
            parameter.numel()
            * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def contains_pruning_masks(
    model: nn.Module,
) -> bool:
    """Modelde maskeli budama kalıntısı olup olmadığını kontrol eder."""

    forbidden_fragments = (
        "weight_orig",
        "weight_mask",
        "bias_orig",
        "bias_mask",
    )

    return any(
        fragment in state_name
        for state_name in model.state_dict()
        for fragment in forbidden_fragments
    )


def _normalize_importance(
    values: Tensor,
) -> Tensor:
    """Önem değerlerini ortalaması 1 olacak biçimde normalleştirir."""

    values = (
        values.detach()
        .float()
    )

    if values.ndim != 1:
        raise ValueError(
            "Önem dizisi tek boyutlu olmalıdır."
        )

    if not torch.isfinite(
        values
    ).all():
        raise ValueError(
            "Önem değerlerinde NaN veya Inf bulundu."
        )

    mean_value = values.mean()

    if float(
        mean_value.item()
    ) <= 0.0:
        return torch.ones_like(
            values
        )

    return values / mean_value


def _select_top_indices(
    importance_scores: Tensor,
    keep_count: int,
) -> Tensor:
    """
    En yüksek öneme sahip indisleri deterministik biçimde seçer.

    Eşit skor durumunda küçük özgün indis önce seçilir.
    """

    score_values = (
        importance_scores.detach()
        .cpu()
        .double()
        .numpy()
    )

    if score_values.ndim != 1:
        raise ValueError(
            "Önem skorları tek boyutlu olmalıdır."
        )

    if not np.isfinite(
        score_values
    ).all():
        raise ValueError(
            "Önem skorlarında sonlu olmayan değer var."
        )

    if not 1 <= keep_count <= len(
        score_values
    ):
        raise ValueError(
            "Korunacak birim sayısı geçersiz."
        )

    original_indices = np.arange(
        len(score_values),
        dtype=np.int64,
    )

    ranking = np.lexsort(
        (
            original_indices,
            -score_values,
        )
    )

    selected_indices = np.sort(
        ranking[:keep_count]
    )

    return torch.as_tensor(
        selected_indices,
        dtype=torch.long,
    )


def _linear_bidirectional_importance(
    current_layer: nn.Linear,
    next_layer: nn.Linear,
) -> Tensor:
    """Linear nöronlarının gelen ve giden L1 önemini birleştirir."""

    incoming_importance = (
        current_layer.weight.detach()
        .abs()
        .sum(dim=1)
    )

    outgoing_importance = (
        next_layer.weight.detach()
        .abs()
        .sum(dim=0)
    )

    if (
        incoming_importance.shape
        != outgoing_importance.shape
    ):
        raise RuntimeError(
            "Linear önem boyutları uyuşmuyor."
        )

    return (
        _normalize_importance(
            incoming_importance
        )
        + _normalize_importance(
            outgoing_importance
        )
    )


def _conv_bidirectional_importance(
    current_layer: nn.Conv1d,
    next_layer: nn.Conv1d | nn.Linear,
) -> Tensor:
    """Conv1d kanallarının gelen ve giden L1 önemini birleştirir."""

    incoming_importance = (
        current_layer.weight.detach()
        .abs()
        .sum(
            dim=(
                1,
                2,
            )
        )
    )

    if isinstance(
        next_layer,
        nn.Conv1d,
    ):
        if next_layer.groups != 1:
            raise ValueError(
                "Gruplu Conv1d budaması desteklenmiyor."
            )

        outgoing_importance = (
            next_layer.weight.detach()
            .abs()
            .sum(
                dim=(
                    0,
                    2,
                )
            )
        )

    elif isinstance(
        next_layer,
        nn.Linear,
    ):
        outgoing_importance = (
            next_layer.weight.detach()
            .abs()
            .sum(dim=0)
        )

    else:
        raise TypeError(
            "Sonraki katman Conv1d veya Linear olmalıdır."
        )

    if (
        incoming_importance.shape
        != outgoing_importance.shape
    ):
        raise RuntimeError(
            "Conv1d önem boyutları uyuşmuyor."
        )

    return (
        _normalize_importance(
            incoming_importance
        )
        + _normalize_importance(
            outgoing_importance
        )
    )


def _copy_linear(
    *,
    source: nn.Linear,
    destination: nn.Linear,
    output_indices: Tensor | None,
    input_indices: Tensor | None,
) -> None:
    """Linear ağırlıklarını seçilmiş indislerle kopyalar."""

    source_weight = (
        source.weight.detach()
    )

    if output_indices is not None:
        output_indices = output_indices.to(
            source_weight.device
        )

        source_weight = source_weight[
            output_indices,
            :,
        ]

    if input_indices is not None:
        input_indices = input_indices.to(
            source_weight.device
        )

        source_weight = source_weight[
            :,
            input_indices,
        ]

    if tuple(
        source_weight.shape
    ) != tuple(
        destination.weight.shape
    ):
        raise RuntimeError(
            "Linear ağırlık boyutları uyuşmuyor: "
            f"kaynak={tuple(source_weight.shape)}, "
            f"hedef={tuple(destination.weight.shape)}"
        )

    with torch.no_grad():
        destination.weight.copy_(
            source_weight.to(
                dtype=destination.weight.dtype,
                device=destination.weight.device,
            )
        )

        if destination.bias is not None:
            if source.bias is None:
                raise RuntimeError(
                    "Kaynak Linear bias içermiyor."
                )

            source_bias = (
                source.bias.detach()
            )

            if output_indices is not None:
                source_bias = source_bias[
                    output_indices
                ]

            if tuple(
                source_bias.shape
            ) != tuple(
                destination.bias.shape
            ):
                raise RuntimeError(
                    "Linear bias boyutları uyuşmuyor."
                )

            destination.bias.copy_(
                source_bias.to(
                    dtype=destination.bias.dtype,
                    device=destination.bias.device,
                )
            )


def _copy_conv1d(
    *,
    source: nn.Conv1d,
    destination: nn.Conv1d,
    output_indices: Tensor,
    input_indices: Tensor | None,
) -> None:
    """Conv1d ağırlıklarını seçilmiş kanallarla kopyalar."""

    if (
        source.groups != 1
        or destination.groups != 1
    ):
        raise ValueError(
            "Gruplu Conv1d kopyalama desteklenmiyor."
        )

    source_weight = (
        source.weight.detach()
    )

    output_indices = output_indices.to(
        source_weight.device
    )

    source_weight = source_weight[
        output_indices,
        :,
        :,
    ]

    if input_indices is not None:
        input_indices = input_indices.to(
            source_weight.device
        )

        source_weight = source_weight[
            :,
            input_indices,
            :,
        ]

    if tuple(
        source_weight.shape
    ) != tuple(
        destination.weight.shape
    ):
        raise RuntimeError(
            "Conv1d ağırlık boyutları uyuşmuyor: "
            f"kaynak={tuple(source_weight.shape)}, "
            f"hedef={tuple(destination.weight.shape)}"
        )

    with torch.no_grad():
        destination.weight.copy_(
            source_weight.to(
                dtype=destination.weight.dtype,
                device=destination.weight.device,
            )
        )

        if destination.bias is not None:
            if source.bias is None:
                raise RuntimeError(
                    "Kaynak Conv1d bias içermiyor."
                )

            source_bias = (
                source.bias.detach()[
                    output_indices
                ]
            )

            if tuple(
                source_bias.shape
            ) != tuple(
                destination.bias.shape
            ):
                raise RuntimeError(
                    "Conv1d bias boyutları uyuşmuyor."
                )

            destination.bias.copy_(
                source_bias.to(
                    dtype=destination.bias.dtype,
                    device=destination.bias.device,
                )
            )


def _calculate_mlp_macs(
    input_features: int,
    hidden_units: tuple[int, ...],
    num_classes: int,
) -> int:
    """MLP/DNN ağırlık katmanlarının MAC sayısını hesaplar."""

    dimensions = (
        input_features,
        *hidden_units,
        num_classes,
    )

    return int(
        sum(
            dimensions[index]
            * dimensions[index + 1]
            for index in range(
                len(dimensions) - 1
            )
        )
    )


def _conv_output_length(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    """Conv/pooling çıktı uzunluğunu hesaplar."""

    return int(
        math.floor(
            (
                input_length
                + 2 * padding
                - dilation
                * (
                    kernel_size - 1
                )
                - 1
            )
            / stride
            + 1
        )
    )


def _calculate_cnn_macs(
    input_features: int,
    channels: tuple[int, int, int],
    num_classes: int,
) -> int:
    """Tiny-1D-CNN ağırlık katmanlarının MAC sayısını hesaplar."""

    channel_1, channel_2, channel_3 = (
        channels
    )

    conv_1_length = _conv_output_length(
        input_length=input_features,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    pool_1_length = _conv_output_length(
        input_length=conv_1_length,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    conv_2_length = _conv_output_length(
        input_length=pool_1_length,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    pool_2_length = _conv_output_length(
        input_length=conv_2_length,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    conv_3_length = _conv_output_length(
        input_length=pool_2_length,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    conv_1_macs = (
        channel_1
        * conv_1_length
        * 1
        * 5
    )

    conv_2_macs = (
        channel_2
        * conv_2_length
        * channel_1
        * 3
    )

    conv_3_macs = (
        channel_3
        * conv_3_length
        * channel_2
        * 3
    )

    classifier_macs = (
        channel_3
        * num_classes
    )

    return int(
        conv_1_macs
        + conv_2_macs
        + conv_3_macs
        + classifier_macs
    )


def _build_metadata(
    *,
    model_name: str,
    pruning_ratio: float,
    input_features: int,
    num_classes: int,
    original_architecture: dict[str, Any],
    compact_architecture: dict[str, Any],
    selected_indices: dict[str, list[int]],
    original_model: nn.Module,
    compact_model: nn.Module,
    original_macs: int,
    compact_macs: int,
) -> StructuredPruningMetadata:
    """Budama metadata kaydını oluşturur."""

    original_parameter_count = (
        count_model_parameters(
            original_model
        )
    )

    compact_parameter_count = (
        count_model_parameters(
            compact_model
        )
    )

    removed_parameter_count = (
        original_parameter_count
        - compact_parameter_count
    )

    removed_macs = (
        original_macs
        - compact_macs
    )

    if removed_parameter_count <= 0:
        raise RuntimeError(
            "Budama parametre sayısını azaltmadı."
        )

    if removed_macs <= 0:
        raise RuntimeError(
            "Budama MAC sayısını azaltmadı."
        )

    return StructuredPruningMetadata(
        model_name=model_name,
        requested_pruning_ratio=float(
            pruning_ratio
        ),
        pruning_name=(
            f"P{int(round(pruning_ratio * 100))}"
        ),
        input_features=int(
            input_features
        ),
        num_classes=int(
            num_classes
        ),
        original_architecture=(
            original_architecture
        ),
        compact_architecture=(
            compact_architecture
        ),
        selected_indices=(
            selected_indices
        ),
        original_parameter_count=int(
            original_parameter_count
        ),
        compact_parameter_count=int(
            compact_parameter_count
        ),
        removed_parameter_count=int(
            removed_parameter_count
        ),
        parameter_reduction_ratio=float(
            removed_parameter_count
            / original_parameter_count
        ),
        original_weight_layer_macs=int(
            original_macs
        ),
        compact_weight_layer_macs=int(
            compact_macs
        ),
        removed_weight_layer_macs=int(
            removed_macs
        ),
        mac_reduction_ratio=float(
            removed_macs
            / original_macs
        ),
        importance_method=(
            "Normalized incoming plus outgoing L1 magnitude"
        ),
        physical_compaction=True,
        mask_based_pruning=False,
        output_class_units_pruned=False,
    )


def _prune_tinyml_mlp(
    original_model: nn.Module,
    pruning_ratio: float,
) -> tuple[
    nn.Module,
    StructuredPruningMetadata,
]:
    """TinyML-MLP gizli nöronlarını budar."""

    layer_1 = original_model.network[0]
    layer_2 = original_model.network[2]
    output_layer = original_model.network[4]

    if not all(
        isinstance(
            layer,
            nn.Linear,
        )
        for layer in (
            layer_1,
            layer_2,
            output_layer,
        )
    ):
        raise TypeError(
            "TinyML-MLP katman yapısı beklenen biçimde değil."
        )

    input_features = int(
        layer_1.in_features
    )

    num_classes = int(
        output_layer.out_features
    )

    original_hidden_units = (
        int(layer_1.out_features),
        int(layer_2.out_features),
    )

    selected_1 = _select_top_indices(
        _linear_bidirectional_importance(
            layer_1,
            layer_2,
        ),
        calculate_keep_count(
            original_hidden_units[0],
            pruning_ratio,
        ),
    )

    selected_2 = _select_top_indices(
        _linear_bidirectional_importance(
            layer_2,
            output_layer,
        ),
        calculate_keep_count(
            original_hidden_units[1],
            pruning_ratio,
        ),
    )

    compact_hidden_units = (
        len(selected_1),
        len(selected_2),
    )

    compact_model = create_model(
        model_name="tinyml_mlp",
        input_features=input_features,
        num_classes=num_classes,
        hidden_units=compact_hidden_units,
    )

    _copy_linear(
        source=layer_1,
        destination=compact_model.network[0],
        output_indices=selected_1,
        input_indices=None,
    )

    _copy_linear(
        source=layer_2,
        destination=compact_model.network[2],
        output_indices=selected_2,
        input_indices=selected_1,
    )

    _copy_linear(
        source=output_layer,
        destination=compact_model.network[4],
        output_indices=None,
        input_indices=selected_2,
    )

    metadata = _build_metadata(
        model_name="tinyml_mlp",
        pruning_ratio=pruning_ratio,
        input_features=input_features,
        num_classes=num_classes,
        original_architecture={
            "hidden_units": list(
                original_hidden_units
            ),
        },
        compact_architecture={
            "hidden_units": list(
                compact_hidden_units
            ),
        },
        selected_indices={
            "hidden_1": selected_1.tolist(),
            "hidden_2": selected_2.tolist(),
        },
        original_model=original_model,
        compact_model=compact_model,
        original_macs=_calculate_mlp_macs(
            input_features=input_features,
            hidden_units=original_hidden_units,
            num_classes=num_classes,
        ),
        compact_macs=_calculate_mlp_macs(
            input_features=input_features,
            hidden_units=compact_hidden_units,
            num_classes=num_classes,
        ),
    )

    return compact_model, metadata


def _prune_compact_dnn(
    original_model: nn.Module,
    pruning_ratio: float,
) -> tuple[
    nn.Module,
    StructuredPruningMetadata,
]:
    """Compact-DNN gizli nöronlarını budar."""

    layer_1 = original_model.network[0]
    layer_2 = original_model.network[3]
    layer_3 = original_model.network[6]
    output_layer = original_model.network[8]

    if not all(
        isinstance(
            layer,
            nn.Linear,
        )
        for layer in (
            layer_1,
            layer_2,
            layer_3,
            output_layer,
        )
    ):
        raise TypeError(
            "Compact-DNN katman yapısı beklenen biçimde değil."
        )

    input_features = int(
        layer_1.in_features
    )

    num_classes = int(
        output_layer.out_features
    )

    original_hidden_units = (
        int(layer_1.out_features),
        int(layer_2.out_features),
        int(layer_3.out_features),
    )

    selected_1 = _select_top_indices(
        _linear_bidirectional_importance(
            layer_1,
            layer_2,
        ),
        calculate_keep_count(
            original_hidden_units[0],
            pruning_ratio,
        ),
    )

    selected_2 = _select_top_indices(
        _linear_bidirectional_importance(
            layer_2,
            layer_3,
        ),
        calculate_keep_count(
            original_hidden_units[1],
            pruning_ratio,
        ),
    )

    selected_3 = _select_top_indices(
        _linear_bidirectional_importance(
            layer_3,
            output_layer,
        ),
        calculate_keep_count(
            original_hidden_units[2],
            pruning_ratio,
        ),
    )

    compact_hidden_units = (
        len(selected_1),
        len(selected_2),
        len(selected_3),
    )

    compact_model = create_model(
        model_name="compact_dnn",
        input_features=input_features,
        num_classes=num_classes,
        hidden_units=compact_hidden_units,
        dropout_probability=float(
            original_model.dropout_probability
        ),
    )

    _copy_linear(
        source=layer_1,
        destination=compact_model.network[0],
        output_indices=selected_1,
        input_indices=None,
    )

    _copy_linear(
        source=layer_2,
        destination=compact_model.network[3],
        output_indices=selected_2,
        input_indices=selected_1,
    )

    _copy_linear(
        source=layer_3,
        destination=compact_model.network[6],
        output_indices=selected_3,
        input_indices=selected_2,
    )

    _copy_linear(
        source=output_layer,
        destination=compact_model.network[8],
        output_indices=None,
        input_indices=selected_3,
    )

    metadata = _build_metadata(
        model_name="compact_dnn",
        pruning_ratio=pruning_ratio,
        input_features=input_features,
        num_classes=num_classes,
        original_architecture={
            "hidden_units": list(
                original_hidden_units
            ),
            "dropout_probability": float(
                original_model.dropout_probability
            ),
        },
        compact_architecture={
            "hidden_units": list(
                compact_hidden_units
            ),
            "dropout_probability": float(
                original_model.dropout_probability
            ),
        },
        selected_indices={
            "hidden_1": selected_1.tolist(),
            "hidden_2": selected_2.tolist(),
            "hidden_3": selected_3.tolist(),
        },
        original_model=original_model,
        compact_model=compact_model,
        original_macs=_calculate_mlp_macs(
            input_features=input_features,
            hidden_units=original_hidden_units,
            num_classes=num_classes,
        ),
        compact_macs=_calculate_mlp_macs(
            input_features=input_features,
            hidden_units=compact_hidden_units,
            num_classes=num_classes,
        ),
    )

    return compact_model, metadata


def _prune_tiny_1d_cnn(
    original_model: nn.Module,
    pruning_ratio: float,
) -> tuple[
    nn.Module,
    StructuredPruningMetadata,
]:
    """Tiny-1D-CNN çıktı kanallarını budar."""

    conv_1 = original_model.feature_extractor[0]
    conv_2 = original_model.feature_extractor[3]
    conv_3 = original_model.feature_extractor[6]
    output_layer = original_model.classifier

    if not (
        isinstance(conv_1, nn.Conv1d)
        and isinstance(conv_2, nn.Conv1d)
        and isinstance(conv_3, nn.Conv1d)
        and isinstance(output_layer, nn.Linear)
    ):
        raise TypeError(
            "Tiny-1D-CNN katman yapısı beklenen biçimde değil."
        )

    input_features = int(
        original_model.input_features
    )

    num_classes = int(
        output_layer.out_features
    )

    original_channels = (
        int(conv_1.out_channels),
        int(conv_2.out_channels),
        int(conv_3.out_channels),
    )

    selected_1 = _select_top_indices(
        _conv_bidirectional_importance(
            conv_1,
            conv_2,
        ),
        calculate_keep_count(
            original_channels[0],
            pruning_ratio,
        ),
    )

    selected_2 = _select_top_indices(
        _conv_bidirectional_importance(
            conv_2,
            conv_3,
        ),
        calculate_keep_count(
            original_channels[1],
            pruning_ratio,
        ),
    )

    selected_3 = _select_top_indices(
        _conv_bidirectional_importance(
            conv_3,
            output_layer,
        ),
        calculate_keep_count(
            original_channels[2],
            pruning_ratio,
        ),
    )

    compact_channels = (
        len(selected_1),
        len(selected_2),
        len(selected_3),
    )

    compact_model = create_model(
        model_name="tiny_1d_cnn",
        input_features=input_features,
        num_classes=num_classes,
        channels=compact_channels,
    )

    _copy_conv1d(
        source=conv_1,
        destination=compact_model.feature_extractor[0],
        output_indices=selected_1,
        input_indices=None,
    )

    _copy_conv1d(
        source=conv_2,
        destination=compact_model.feature_extractor[3],
        output_indices=selected_2,
        input_indices=selected_1,
    )

    _copy_conv1d(
        source=conv_3,
        destination=compact_model.feature_extractor[6],
        output_indices=selected_3,
        input_indices=selected_2,
    )

    _copy_linear(
        source=output_layer,
        destination=compact_model.classifier,
        output_indices=None,
        input_indices=selected_3,
    )

    metadata = _build_metadata(
        model_name="tiny_1d_cnn",
        pruning_ratio=pruning_ratio,
        input_features=input_features,
        num_classes=num_classes,
        original_architecture={
            "channels": list(
                original_channels
            ),
        },
        compact_architecture={
            "channels": list(
                compact_channels
            ),
        },
        selected_indices={
            "conv_1": selected_1.tolist(),
            "conv_2": selected_2.tolist(),
            "conv_3": selected_3.tolist(),
        },
        original_model=original_model,
        compact_model=compact_model,
        original_macs=_calculate_cnn_macs(
            input_features=input_features,
            channels=original_channels,
            num_classes=num_classes,
        ),
        compact_macs=_calculate_cnn_macs(
            input_features=input_features,
            channels=compact_channels,
            num_classes=num_classes,
        ),
    )

    return compact_model, metadata


def physically_prune_model(
    *,
    model_name: str,
    original_model: nn.Module,
    pruning_ratio: float,
) -> tuple[
    nn.Module,
    StructuredPruningMetadata,
]:
    """FP32 modeli fiziksel olarak yapılandırılmış biçimde budar."""

    if model_name not in SUPPORTED_MODELS:
        raise KeyError(
            f"Budama desteklenmiyor: {model_name}"
        )

    ratio = validate_pruning_ratio(
        pruning_ratio
    )

    original_model = (
        original_model.cpu()
    )

    if model_name == "tinyml_mlp":
        return _prune_tinyml_mlp(
            original_model=original_model,
            pruning_ratio=ratio,
        )

    if model_name == "compact_dnn":
        return _prune_compact_dnn(
            original_model=original_model,
            pruning_ratio=ratio,
        )

    if model_name == "tiny_1d_cnn":
        return _prune_tiny_1d_cnn(
            original_model=original_model,
            pruning_ratio=ratio,
        )

    raise RuntimeError(
        "Ulaşılamayan budama dalı."
    )


def recreate_compact_model(
    pruning_metadata: dict[str, Any],
) -> nn.Module:
    """Budama metadata kaydından kompakt model oluşturur."""

    model_name = str(
        pruning_metadata[
            "model_name"
        ]
    )

    input_features = int(
        pruning_metadata[
            "input_features"
        ]
    )

    num_classes = int(
        pruning_metadata[
            "num_classes"
        ]
    )

    compact_architecture = (
        pruning_metadata[
            "compact_architecture"
        ]
    )

    if model_name == "tinyml_mlp":
        return create_model(
            model_name=model_name,
            input_features=input_features,
            num_classes=num_classes,
            hidden_units=tuple(
                int(value)
                for value in compact_architecture[
                    "hidden_units"
                ]
            ),
        )

    if model_name == "compact_dnn":
        return create_model(
            model_name=model_name,
            input_features=input_features,
            num_classes=num_classes,
            hidden_units=tuple(
                int(value)
                for value in compact_architecture[
                    "hidden_units"
                ]
            ),
            dropout_probability=float(
                compact_architecture[
                    "dropout_probability"
                ]
            ),
        )

    if model_name == "tiny_1d_cnn":
        return create_model(
            model_name=model_name,
            input_features=input_features,
            num_classes=num_classes,
            channels=tuple(
                int(value)
                for value in compact_architecture[
                    "channels"
                ]
            ),
        )

    raise KeyError(
        f"Metadata modeli desteklenmiyor: {model_name}"
    )