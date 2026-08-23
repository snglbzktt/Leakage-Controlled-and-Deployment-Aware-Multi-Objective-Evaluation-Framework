from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

IMPORTANCE_CRITERION = (
    "For hidden unit j: "
    "score_j = L2([incoming weight row j, bias j]) "
    "* L2(outgoing weight column j). "
    "Scores are computed independently on the original B0 checkpoint. "
    "Ranking is descending by score, with ascending original unit index "
    "as the deterministic tie-break. P25 and P50 are each generated "
    "directly from the same B0 checkpoint."
)


def _named_linear_layers(
    model: nn.Module,
) -> list[tuple[str, nn.Linear]]:
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]

    if len(layers) < 2:
        raise ValueError(
            "At least one hidden Linear layer and one output "
            "Linear layer are required."
        )

    return layers


def _validate_linear_chain(
    layers: list[tuple[str, nn.Linear]],
) -> None:
    for index in range(len(layers) - 1):
        current = layers[index][1]
        following = layers[index + 1][1]

        if current.out_features != following.in_features:
            raise ValueError(
                "Linear layers are not a simple feed-forward chain: "
                f"{layers[index][0]} -> {layers[index + 1][0]}"
            )


def _set_submodule(
    root: nn.Module,
    target: str,
    replacement: nn.Module,
) -> None:
    parts = target.split(".")
    parent = root

    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(
                f"Cannot resolve module path: {target}"
            )
        parent = parent._modules[part]

    leaf = parts[-1]

    if leaf not in parent._modules:
        raise KeyError(
            f"Cannot resolve module path: {target}"
        )

    parent._modules[leaf] = replacement


def _retained_count(
    original_width: int,
    pruning_ratio: float,
) -> int:
    retained = max(
        1,
        round(
            (1.0 - pruning_ratio) * original_width
        ),
    )

    if retained >= original_width:
        raise ValueError(
            "Pruning ratio did not reduce the hidden width."
        )

    return int(retained)


def _importance_scores(
    incoming: nn.Linear,
    outgoing: nn.Linear,
) -> np.ndarray:
    incoming_weight = (
        incoming.weight.detach()
        .to(device="cpu", dtype=torch.float64)
    )

    incoming_energy = torch.sum(
        incoming_weight.square(),
        dim=1,
    )

    if incoming.bias is not None:
        incoming_bias = (
            incoming.bias.detach()
            .to(device="cpu", dtype=torch.float64)
        )

        incoming_energy = (
            incoming_energy
            + incoming_bias.square()
        )

    outgoing_weight = (
        outgoing.weight.detach()
        .to(device="cpu", dtype=torch.float64)
    )

    outgoing_energy = torch.sum(
        outgoing_weight.square(),
        dim=0,
    )

    scores = torch.sqrt(
        incoming_energy
    ) * torch.sqrt(
        outgoing_energy
    )

    if not torch.isfinite(scores).all():
        raise ValueError(
            "Non-finite hidden-unit importance score."
        )

    return scores.numpy()


def _deterministic_selection(
    scores: np.ndarray,
    retained_count: int,
) -> tuple[list[int], list[int]]:
    if scores.ndim != 1:
        raise ValueError(
            "Importance scores must be one-dimensional."
        )

    if not (
        1 <= retained_count <= len(scores)
    ):
        raise ValueError(
            "Invalid retained-unit count."
        )

    indices = np.arange(
        len(scores),
        dtype=np.int64,
    )

    ranking = np.lexsort(
        (
            indices,
            -scores,
        )
    )

    ranked_indices = [
        int(value)
        for value in ranking.tolist()
    ]

    selected_indices = sorted(
        ranked_indices[:retained_count]
    )

    return selected_indices, ranked_indices


def inspect_linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(layer.in_features),
            int(layer.out_features),
        ]
        for _, layer in _named_linear_layers(model)
    ]


def count_parameters(
    model: nn.Module,
) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def linear_macs_per_sample(
    model: nn.Module,
) -> int:
    return int(
        sum(
            layer.in_features * layer.out_features
            for _, layer in _named_linear_layers(model)
        )
    )


def physically_prune_mlp(
    model: nn.Module,
    pruning_ratio: float,
    *,
    source_checkpoint_sha256: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be torch.nn.Module."
        )

    pruning_ratio = float(pruning_ratio)

    if not (
        0.0 < pruning_ratio < 1.0
    ):
        raise ValueError(
            "pruning_ratio must be strictly between 0 and 1."
        )

    if not source_checkpoint_sha256:
        raise ValueError(
            "source_checkpoint_sha256 is required."
        )

    source_layers = _named_linear_layers(model)
    _validate_linear_chain(source_layers)

    retained_indices_by_hidden_layer: list[list[int]] = []
    selections: list[dict[str, Any]] = []

    for hidden_index, (
        (layer_name, incoming),
        (_, outgoing),
    ) in enumerate(
        zip(
            source_layers[:-1],
            source_layers[1:],
        )
    ):
        scores = _importance_scores(
            incoming,
            outgoing,
        )

        retained_count = _retained_count(
            incoming.out_features,
            pruning_ratio,
        )

        (
            selected_indices,
            ranked_indices,
        ) = _deterministic_selection(
            scores,
            retained_count,
        )

        retained_indices_by_hidden_layer.append(
            selected_indices
        )

        selections.append(
            {
                "hidden_layer_index": hidden_index,
                "layer_name": layer_name,
                "original_width": int(
                    incoming.out_features
                ),
                "retained_width": retained_count,
                "selected_indices": selected_indices,
                "ranked_indices": ranked_indices,
                "selected_scores": [
                    float(scores[index])
                    for index in selected_indices
                ],
            }
        )

    pruned_model = copy.deepcopy(model)
    pruned_model.train(model.training)

    layer_copy_records: list[dict[str, Any]] = []

    for layer_index, (
        layer_name,
        source_layer,
    ) in enumerate(source_layers):
        if layer_index == 0:
            input_indices = list(
                range(source_layer.in_features)
            )
        else:
            input_indices = (
                retained_indices_by_hidden_layer[
                    layer_index - 1
                ]
            )

        if layer_index < len(
            retained_indices_by_hidden_layer
        ):
            output_indices = (
                retained_indices_by_hidden_layer[
                    layer_index
                ]
            )
        else:
            output_indices = list(
                range(source_layer.out_features)
            )

        replacement = nn.Linear(
            in_features=len(input_indices),
            out_features=len(output_indices),
            bias=source_layer.bias is not None,
            device=source_layer.weight.device,
            dtype=source_layer.weight.dtype,
        )

        input_tensor = torch.tensor(
            input_indices,
            dtype=torch.long,
            device=source_layer.weight.device,
        )

        output_tensor = torch.tensor(
            output_indices,
            dtype=torch.long,
            device=source_layer.weight.device,
        )

        with torch.no_grad():
            replacement.weight.copy_(
                source_layer.weight
                .index_select(0, output_tensor)
                .index_select(1, input_tensor)
            )

            if source_layer.bias is not None:
                replacement.bias.copy_(
                    source_layer.bias.index_select(
                        0,
                        output_tensor,
                    )
                )

        _set_submodule(
            pruned_model,
            layer_name,
            replacement,
        )

        layer_copy_records.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "input_indices": [
                    int(value)
                    for value in input_indices
                ],
                "output_indices": [
                    int(value)
                    for value in output_indices
                ],
                "source_shape": [
                    int(source_layer.out_features),
                    int(source_layer.in_features),
                ],
                "compacted_shape": [
                    len(output_indices),
                    len(input_indices),
                ],
            }
        )

    metadata: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "importance_criterion": IMPORTANCE_CRITERION,
        "pruning_ratio": pruning_ratio,
        "source_checkpoint_sha256": (
            source_checkpoint_sha256
        ),
        "source_linear_widths": (
            inspect_linear_widths(model)
        ),
        "compacted_linear_widths": (
            inspect_linear_widths(pruned_model)
        ),
        "source_parameter_count": (
            count_parameters(model)
        ),
        "compacted_parameter_count": (
            count_parameters(pruned_model)
        ),
        "source_linear_macs_per_sample": (
            linear_macs_per_sample(model)
        ),
        "compacted_linear_macs_per_sample": (
            linear_macs_per_sample(pruned_model)
        ),
        "hidden_unit_selections": selections,
        "layer_copy_records": layer_copy_records,
        "physical_compaction": True,
        "mask_only_pruning": False,
        "generated_directly_from_B0": True,
    }

    if (
        metadata["compacted_parameter_count"]
        >= metadata["source_parameter_count"]
    ):
        raise RuntimeError(
            "Physical pruning did not reduce parameter count."
        )

    if (
        metadata[
            "compacted_linear_macs_per_sample"
        ]
        >= metadata[
            "source_linear_macs_per_sample"
        ]
    ):
        raise RuntimeError(
            "Physical pruning did not reduce linear MACs."
        )

    return pruned_model, metadata
