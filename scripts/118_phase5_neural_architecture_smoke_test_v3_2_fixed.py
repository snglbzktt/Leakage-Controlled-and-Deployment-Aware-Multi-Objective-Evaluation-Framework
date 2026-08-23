from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"
Y_TRAIN_PATH = FINAL_CACHE / "y_train.npy"

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

PHASE5_PROTOCOL_COMPLETION = (
    AUDIT
    / "phase5_protocol_locked_v3_2.json"
)

PHASE5_PROTOCOL_LOCK = (
    AUDIT
    / "phase5_protocol_lock_manifest_v3_2.json"
)

PREPROCESSING_DIRECTORY = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
)

SCALER_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

CLASS_WEIGHTS_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.npz"
)

PREPROCESSING_COMPLETION = (
    AUDIT
    / "phase5_preprocessing_locked_v3_2.json"
)

PREPROCESSING_VERIFICATION = (
    AUDIT
    / "phase5_preprocessing_verification_v3_2.json"
)

PREPROCESSING_LOCK = (
    AUDIT
    / "phase5_preprocessing_lock_manifest_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

MODEL_REGISTRY = (
    ROOT
    / "models"
    / "architecture"
    / "nbaiot_model_registry_v1.json"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase5_neural_architecture_smoke_test_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase5_neural_architecture_smoke_test_locked_v3_2.json"
)

EXPECTED_PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

EXPECTED_DATA_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_TRAIN_ROWS = 1_534_583

SMOKE_BATCH_SIZE = 256
SMOKE_SEED = 2026
CPU_THREAD_COUNT = 4

ARCHITECTURE_SPECS = {
    "tinyml_mlp": {
        "expected_linear_widths": [
            [115, 64],
            [64, 32],
            [32, 3],
        ],
        "hidden_dims": [64, 32],
        "b0_learning_rate": 0.003,
        "name_tokens": (
            "tinymlmlp",
            "tinyml",
        ),
    },
    "compact_dnn": {
        "expected_linear_widths": [
            [115, 128],
            [128, 64],
            [64, 32],
            [32, 3],
        ],
        "hidden_dims": [128, 64, 32],
        "b0_learning_rate": 0.001,
        "name_tokens": (
            "compactdnn",
            "compact",
        ),
    },
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def normalize_name(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_module(
    path: Path,
) -> ModuleType:
    module_name = (
        "phase5_locked_nbaiot_models"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not create an import "
            "specification for the model source."
        )

    module = importlib.util.module_from_spec(
        specification
    )

    # Dataclasses inspect sys.modules while the module is being executed.
    # Register the dynamically loaded module before exec_module.
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


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
        "Model output does not contain "
        "a usable logits tensor."
    )


def linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(layer.in_features),
            int(layer.out_features),
        ]
        for layer in model.modules()
        if isinstance(
            layer,
            nn.Linear,
        )
    ]


def callable_kwargs(
    candidate: Callable[..., Any],
    architecture: str,
) -> dict[str, Any] | None:
    signature = inspect.signature(
        candidate
    )

    hidden_dims = (
        ARCHITECTURE_SPECS[
            architecture
        ]["hidden_dims"]
    )

    kwargs: dict[str, Any] = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in {
            "self",
            "args",
            "kwargs",
        }:
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        normalized = normalize_name(name)

        if normalized in {
            "inputdim",
            "inputsize",
            "inputfeatures",
            "nfeatures",
            "numfeatures",
            "infeatures",
            "featurecount",
        }:
            kwargs[name] = (
                EXPECTED_FEATURE_COUNT
            )
            continue

        if normalized in {
            "numclasses",
            "nclasses",
            "classcount",
            "outputdim",
            "outputsize",
            "outfeatures",
        }:
            kwargs[name] = (
                EXPECTED_CLASS_COUNT
            )
            continue

        if normalized in {
            "hiddendims",
            "hiddensizes",
            "hiddenlayers",
            "hiddenunits",
        }:
            kwargs[name] = list(
                hidden_dims
            )
            continue

        if normalized in {
            "architecture",
            "architecturename",
            "modelname",
            "modelid",
        }:
            kwargs[name] = architecture
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            return None

    return kwargs


def candidate_score(
    name: str,
    architecture: str,
) -> int:
    normalized = normalize_name(name)

    tokens = (
        ARCHITECTURE_SPECS[
            architecture
        ]["name_tokens"]
    )

    score = 0

    for index, token in enumerate(
        tokens
    ):
        if normalized == token:
            score = max(
                score,
                100 - index,
            )
        elif token in normalized:
            score = max(
                score,
                80 - index,
            )

    if any(
        token in normalized
        for token in (
            "build",
            "create",
            "make",
            "model",
            "network",
        )
    ):
        score += 5

    return score


def collect_candidates(
    module: ModuleType,
    architecture: str,
) -> list[
    tuple[int, str, Callable[..., Any]]
]:
    records: list[
        tuple[
            int,
            str,
            Callable[..., Any],
        ]
    ] = []

    for name, value in vars(
        module
    ).items():
        if name.startswith("_"):
            continue

        is_module_class = (
            inspect.isclass(value)
            and issubclass(
                value,
                nn.Module,
            )
            and value is not nn.Module
        )

        is_factory = (
            inspect.isfunction(value)
            and value.__module__
            == module.__name__
        )

        if not (
            is_module_class
            or is_factory
        ):
            continue

        score = candidate_score(
            name,
            architecture,
        )

        if score <= 0:
            continue

        records.append(
            (
                score,
                name,
                value,
            )
        )

    records.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return records


def instantiate_candidate(
    candidate: Callable[..., Any],
    architecture: str,
) -> nn.Module:
    kwargs = callable_kwargs(
        candidate,
        architecture,
    )

    if kwargs is None:
        raise RuntimeError(
            "Required constructor "
            "arguments could not be resolved."
        )

    set_all_seeds(
        SMOKE_SEED
    )

    model = candidate(
        **kwargs
    )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Candidate did not return "
            "torch.nn.Module."
        )

    return model


def resolve_architecture(
    module: ModuleType,
    architecture: str,
    probe_batch: torch.Tensor,
) -> tuple[nn.Module, dict[str, Any]]:
    expected_widths = (
        ARCHITECTURE_SPECS[
            architecture
        ]["expected_linear_widths"]
    )

    candidates = collect_candidates(
        module,
        architecture,
    )

    attempts: list[
        dict[str, Any]
    ] = []

    valid: list[
        tuple[
            int,
            str,
            nn.Module,
            list[list[int]],
        ]
    ] = []

    for score, name, candidate in candidates:
        attempt: dict[str, Any] = {
            "name": name,
            "score": score,
            "status": "failed",
        }

        try:
            model = instantiate_candidate(
                candidate,
                architecture,
            )

            model.eval()

            with torch.no_grad():
                logits = extract_logits(
                    model(
                        probe_batch
                    )
                )

            widths = linear_widths(
                model
            )

            attempt[
                "linear_widths"
            ] = widths

            attempt[
                "logit_shape"
            ] = list(
                logits.shape
            )

            if widths != expected_widths:
                raise RuntimeError(
                    "Linear topology mismatch."
                )

            if tuple(
                logits.shape
            ) != (
                len(probe_batch),
                EXPECTED_CLASS_COUNT,
            ):
                raise RuntimeError(
                    "Logit shape mismatch."
                )

            attempt["status"] = "valid"

            valid.append(
                (
                    score,
                    name,
                    model,
                    widths,
                )
            )

        except Exception as error:
            attempt["error"] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        attempts.append(attempt)

    if len(valid) != 1:
        module_classes = sorted(
            name
            for name, value in vars(
                module
            ).items()
            if inspect.isclass(value)
            and issubclass(
                value,
                nn.Module,
            )
            and value is not nn.Module
        )

        raise RuntimeError(
            "Expected exactly one valid "
            f"{architecture} implementation, "
            f"found {len(valid)}. "
            f"Module nn.Module classes: "
            f"{module_classes}. "
            f"Attempts: {attempts}"
        )

    (
        score,
        selected_name,
        selected_model,
        widths,
    ) = valid[0]

    resolution = {
        "selected_symbol": (
            selected_name
        ),
        "selected_score": score,
        "linear_widths": widths,
        "candidate_attempts": attempts,
        "valid_candidate_count": 1,
    }

    return (
        selected_model,
        resolution,
    )


def state_dict_equal(
    first: nn.Module,
    second: nn.Module,
) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()

    if (
        first_state.keys()
        != second_state.keys()
    ):
        return False

    return all(
        torch.equal(
            first_state[key],
            second_state[key],
        )
        for key in first_state
    )


def clone_trainable_parameters(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value
        in model.named_parameters()
        if value.requires_grad
    }


def changed_parameter_count(
    before: dict[str, torch.Tensor],
    model: nn.Module,
) -> int:
    changed = 0

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if not torch.equal(
            before[name],
            parameter.detach(),
        ):
            changed += 1

    return changed


def gradient_statistics(
    model: nn.Module,
) -> dict[str, float | int]:
    squared_norm = 0.0
    finite = True
    tensor_count = 0
    nonzero_tensor_count = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        tensor_count += 1

        gradient = (
            parameter.grad.detach()
        )

        if not torch.isfinite(
            gradient
        ).all():
            finite = False

        norm = float(
            torch.linalg.vector_norm(
                gradient
            ).item()
        )

        squared_norm += norm * norm

        if norm > 0.0:
            nonzero_tensor_count += 1

    return {
        "gradient_tensor_count": (
            tensor_count
        ),
        "nonzero_gradient_tensor_count": (
            nonzero_tensor_count
        ),
        "global_gradient_norm": (
            math.sqrt(
                squared_norm
            )
        ),
        "all_gradients_finite": finite,
    }


def smoke_test_architecture(
    module: ModuleType,
    architecture: str,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    probe_batch = x_batch[:8]

    model, resolution = (
        resolve_architecture(
            module,
            architecture,
            probe_batch,
        )
    )

    set_all_seeds(
        SMOKE_SEED
    )

    second_model, _ = (
        resolve_architecture(
            module,
            architecture,
            probe_batch,
        )
    )

    deterministic_initialization = (
        state_dict_equal(
            model,
            second_model,
        )
    )

    if not deterministic_initialization:
        raise RuntimeError(
            f"{architecture}: deterministic "
            "initialization check failed."
        )

    expected_widths = (
        ARCHITECTURE_SPECS[
            architecture
        ]["expected_linear_widths"]
    )

    observed_widths = linear_widths(
        model
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    linear_macs_per_sample = sum(
        int(layer.in_features)
        * int(layer.out_features)
        for layer in model.modules()
        if isinstance(
            layer,
            nn.Linear,
        )
    )

    model.train()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ARCHITECTURE_SPECS[
            architecture
        ]["b0_learning_rate"],
        weight_decay=0.0001,
    )

    before = clone_trainable_parameters(
        model
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits_before = extract_logits(
        model(x_batch)
    )

    loss = criterion(
        logits_before,
        y_batch,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"{architecture}: non-finite loss."
        )

    loss.backward()

    gradient_report = (
        gradient_statistics(model)
    )

    optimizer.step()

    changed_count = (
        changed_parameter_count(
            before,
            model,
        )
    )

    model.eval()

    with torch.no_grad():
        logits_after = extract_logits(
            model(x_batch)
        )

        probabilities = (
            torch.softmax(
                logits_after,
                dim=1,
            )
        )

    checks = {
        "linear_topology_matches": (
            observed_widths
            == expected_widths
        ),
        "deterministic_initialization": (
            deterministic_initialization
        ),
        "parameter_count_positive": (
            parameter_count > 0
        ),
        "all_parameters_trainable": (
            trainable_parameter_count
            == parameter_count
        ),
        "training_logit_shape_matches": (
            tuple(
                logits_before.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_CLASS_COUNT,
            )
        ),
        "evaluation_logit_shape_matches": (
            tuple(
                logits_after.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_CLASS_COUNT,
            )
        ),
        "training_logits_finite": (
            bool(
                torch.isfinite(
                    logits_before
                ).all().item()
            )
        ),
        "evaluation_logits_finite": (
            bool(
                torch.isfinite(
                    logits_after
                ).all().item()
            )
        ),
        "loss_finite": (
            bool(
                torch.isfinite(
                    loss
                ).item()
            )
        ),
        "gradient_tensors_present": (
            int(
                gradient_report[
                    "gradient_tensor_count"
                ]
            )
            > 0
        ),
        "nonzero_gradients_present": (
            int(
                gradient_report[
                    "nonzero_gradient_tensor_count"
                ]
            )
            > 0
        ),
        "gradients_finite": (
            bool(
                gradient_report[
                    "all_gradients_finite"
                ]
            )
        ),
        "optimizer_changed_parameters": (
            changed_count > 0
        ),
        "probability_shape_matches": (
            tuple(
                probabilities.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_CLASS_COUNT,
            )
        ),
        "probabilities_finite": (
            bool(
                torch.isfinite(
                    probabilities
                ).all().item()
            )
        ),
        "probabilities_sum_to_one": (
            bool(
                torch.allclose(
                    probabilities.sum(
                        dim=1
                    ),
                    torch.ones(
                        SMOKE_BATCH_SIZE,
                        dtype=(
                            probabilities.dtype
                        ),
                    ),
                    atol=1e-6,
                    rtol=0.0,
                )
            )
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            f"{architecture} smoke test "
            "failed: "
            + ", ".join(
                failed_checks
            )
        )

    return {
        "architecture": architecture,
        "selected_symbol": (
            resolution[
                "selected_symbol"
            ]
        ),
        "linear_widths": (
            observed_widths
        ),
        "parameter_count": int(
            parameter_count
        ),
        "trainable_parameter_count": int(
            trainable_parameter_count
        ),
        "linear_macs_per_sample": int(
            linear_macs_per_sample
        ),
        "smoke_batch_size": (
            SMOKE_BATCH_SIZE
        ),
        "initial_loss": float(
            loss.item()
        ),
        "changed_parameter_tensor_count": int(
            changed_count
        ),
        "gradient_report": (
            gradient_report
        ),
        "resolution": resolution,
        "checks": checks,
        "all_checks_passed": True,
    }


required_paths = (
    FINAL_CACHE,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    PHASE5_PROTOCOL,
    PHASE5_PROTOCOL_COMPLETION,
    PHASE5_PROTOCOL_LOCK,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    PREPROCESSING_COMPLETION,
    PREPROCESSING_VERIFICATION,
    PREPROCESSING_LOCK,
    MODEL_SOURCE,
    MODEL_REGISTRY,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_REPORT,
    OUTPUT_COMPLETION,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 5 neural smoke-test "
            "artifact already exists; "
            "refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

protocol_completion = read_json(
    PHASE5_PROTOCOL_COMPLETION
)

protocol_lock = read_json(
    PHASE5_PROTOCOL_LOCK
)

preprocessing_completion = (
    read_json(
        PREPROCESSING_COMPLETION
    )
)

preprocessing_verification = (
    read_json(
        PREPROCESSING_VERIFICATION
    )
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)

entry_checks = {
    "phase5_protocol_locked": (
        protocol.get("status")
        == "locked"
    ),
    "phase5_protocol_version_matches": (
        protocol.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "phase5_data_protocol_matches": (
        protocol.get(
            "data_protocol_version"
        )
        == EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "protocol_completion_locked": (
        protocol_completion.get(
            "status"
        )
        == "locked"
        and protocol_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        protocol_completion.get(
            "protocol_sha256"
        )
        == sha256_file(
            PHASE5_PROTOCOL
        )
    ),
    "protocol_lock_manifest_locked": (
        protocol_lock.get("status")
        == "locked"
        and protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_locked": (
        preprocessing_completion.get(
            "status"
        )
        == "locked"
        and preprocessing_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_ready_for_B0": (
        preprocessing_completion.get(
            "ready_for_B0_training"
        )
        is True
    ),
    "preprocessing_verification_passed": (
        preprocessing_verification.get(
            "status"
        )
        == "passed"
        and preprocessing_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_verification_hash_matches": (
        preprocessing_completion.get(
            "verification_sha256"
        )
        == sha256_file(
            PREPROCESSING_VERIFICATION
        )
    ),
    "preprocessing_lock_manifest_locked": (
        preprocessing_lock.get("status")
        == "locked"
        and preprocessing_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "model_source_hash_matches_protocol": (
        protocol[
            "architectures"
        ]["model_source"]["sha256"]
        == sha256_file(
            MODEL_SOURCE
        )
    ),
    "model_registry_hash_matches_protocol": (
        protocol[
            "architectures"
        ]["model_registry"]["sha256"]
        == sha256_file(
            MODEL_REGISTRY
        )
    ),
    "legacy_checkpoint_reuse_disabled": (
        protocol[
            "legacy_reuse_policy"
        ][
            "legacy_checkpoints_directly_reused"
        ]
        is False
    ),
    "test_not_used_for_selection": (
        protocol[
            "test_isolation"
        ]["test_used_for_selection"]
        is False
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 5 neural smoke-test "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

torch.set_num_threads(
    CPU_THREAD_COUNT
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

y_train = np.load(
    Y_TRAIN_PATH,
    mmap_mode="r",
)

input_checks = {
    "x_train_shape_matches": (
        x_train.shape
        == (
            EXPECTED_TRAIN_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "x_train_dtype_float32": (
        x_train.dtype
        == np.float32
    ),
    "y_train_shape_matches": (
        y_train.shape
        == (
            EXPECTED_TRAIN_ROWS,
        )
    ),
}

failed_input_checks = [
    name
    for name, passed
    in input_checks.items()
    if not passed
]

if failed_input_checks:
    raise RuntimeError(
        "Phase 5 smoke-test input "
        "checks failed: "
        + ", ".join(
            failed_input_checks
        )
    )

with np.load(
    SCALER_NPZ
) as values:
    scaler_mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scaler_scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

with np.load(
    CLASS_WEIGHTS_NPZ
) as values:
    class_labels = np.array(
        values["class_labels"],
        copy=True,
    )

    class_weights32 = np.array(
        values["class_weights_float32"],
        copy=True,
    )

preprocessing_checks = {
    "scaler_mean_shape_matches": (
        scaler_mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "scaler_scale_shape_matches": (
        scaler_scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "scaler_values_finite": (
        bool(
            np.isfinite(
                scaler_mean64
            ).all()
            and np.isfinite(
                scaler_scale64
            ).all()
        )
    ),
    "scaler_scale_positive": (
        bool(
            np.all(
                scaler_scale64 > 0.0
            )
        )
    ),
    "class_labels_match": (
        np.array_equal(
            class_labels,
            np.asarray(
                [0, 1, 2],
                dtype=np.int64,
            ),
        )
    ),
    "class_weights_shape_matches": (
        class_weights32.shape
        == (3,)
    ),
    "class_weights_finite_positive": (
        bool(
            np.isfinite(
                class_weights32
            ).all()
            and np.all(
                class_weights32 > 0.0
            )
        )
    ),
}

failed_preprocessing_checks = [
    name
    for name, passed
    in preprocessing_checks.items()
    if not passed
]

if failed_preprocessing_checks:
    raise RuntimeError(
        "Phase 5 smoke-test preprocessing "
        "checks failed: "
        + ", ".join(
            failed_preprocessing_checks
        )
    )

x_batch64 = np.asarray(
    x_train[
        :SMOKE_BATCH_SIZE
    ],
    dtype=np.float64,
)

x_scaled32 = (
    (
        x_batch64
        - scaler_mean64
    )
    / scaler_scale64
).astype(np.float32)

y_batch_array = np.asarray(
    y_train[
        :SMOKE_BATCH_SIZE
    ],
    dtype=np.int64,
)

batch_checks = {
    "scaled_batch_shape_matches": (
        x_scaled32.shape
        == (
            SMOKE_BATCH_SIZE,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "scaled_batch_dtype_float32": (
        x_scaled32.dtype
        == np.float32
    ),
    "scaled_batch_finite": (
        bool(
            np.isfinite(
                x_scaled32
            ).all()
        )
    ),
    "label_batch_shape_matches": (
        y_batch_array.shape
        == (
            SMOKE_BATCH_SIZE,
        )
    ),
    "label_batch_in_range": (
        bool(
            np.all(
                (
                    y_batch_array >= 0
                )
                & (
                    y_batch_array
                    < EXPECTED_CLASS_COUNT
                )
            )
        )
    ),
}

failed_batch_checks = [
    name
    for name, passed
    in batch_checks.items()
    if not passed
]

if failed_batch_checks:
    raise RuntimeError(
        "Phase 5 smoke-test batch "
        "checks failed: "
        + ", ".join(
            failed_batch_checks
        )
    )

x_batch = torch.from_numpy(
    x_scaled32
)

y_batch = torch.from_numpy(
    y_batch_array
)

class_weight_tensor = (
    torch.from_numpy(
        class_weights32
    )
)

module = load_model_module(
    MODEL_SOURCE
)

print("=" * 92)
print("PHASE 5 NEURAL ARCHITECTURE AND ONE-BATCH SMOKE TEST")
print("=" * 92)
print(
    "Device                    : CPU"
)
print(
    "CPU threads               : "
    f"{CPU_THREAD_COUNT}"
)
print(
    "Smoke batch size          : "
    f"{SMOKE_BATCH_SIZE}"
)
print(
    "Input features            : "
    f"{EXPECTED_FEATURE_COUNT}"
)
print(
    "Output classes            : "
    f"{EXPECTED_CLASS_COUNT}"
)
print(
    "Validation access count   : 0"
)
print(
    "Test access count         : 0"
)
print()

architecture_reports = []

for architecture in (
    "tinyml_mlp",
    "compact_dnn",
):
    print(
        f"Testing {architecture}...",
        flush=True,
    )

    report = smoke_test_architecture(
        module=module,
        architecture=architecture,
        x_batch=x_batch,
        y_batch=y_batch,
        class_weights=(
            class_weight_tensor
        ),
    )

    architecture_reports.append(
        report
    )

    print(
        "  symbol="
        f"{report['selected_symbol']} | "
        "parameters="
        f"{report['parameter_count']:,} | "
        "linear MACs/sample="
        f"{report['linear_macs_per_sample']:,} | "
        "loss="
        f"{report['initial_loss']:.6f} | "
        "all checks=True"
    )

cross_architecture_checks = {
    "two_architectures_verified": (
        len(
            architecture_reports
        )
        == 2
    ),
    "architecture_names_match": (
        {
            row["architecture"]
            for row
            in architecture_reports
        }
        == {
            "tinyml_mlp",
            "compact_dnn",
        }
    ),
    "all_architecture_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row
            in architecture_reports
        )
    ),
    "compact_has_more_parameters_than_tinyml": (
        next(
            row["parameter_count"]
            for row
            in architecture_reports
            if row["architecture"]
            == "compact_dnn"
        )
        >
        next(
            row["parameter_count"]
            for row
            in architecture_reports
            if row["architecture"]
            == "tinyml_mlp"
        )
    ),
    "compact_has_more_linear_macs_than_tinyml": (
        next(
            row[
                "linear_macs_per_sample"
            ]
            for row
            in architecture_reports
            if row["architecture"]
            == "compact_dnn"
        )
        >
        next(
            row[
                "linear_macs_per_sample"
            ]
            for row
            in architecture_reports
            if row["architecture"]
            == "tinyml_mlp"
        )
    ),
}

failed_cross_checks = [
    name
    for name, passed
    in cross_architecture_checks.items()
    if not passed
]

if failed_cross_checks:
    raise RuntimeError(
        "Phase 5 cross-architecture "
        "smoke checks failed: "
        + ", ".join(
            failed_cross_checks
        )
    )

all_checks = {
    **entry_checks,
    **input_checks,
    **preprocessing_checks,
    **batch_checks,
    **cross_architecture_checks,
}

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "neural_architecture_one_batch_smoke_test"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "data_protocol_version": (
        EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "device": "cpu",
    "cpu_thread_count": (
        CPU_THREAD_COUNT
    ),
    "torch_version": (
        torch.__version__
    ),
    "smoke_seed": (
        SMOKE_SEED
    ),
    "smoke_batch_size": (
        SMOKE_BATCH_SIZE
    ),
    "fit_scope": {
        "train_rows_accessed": (
            SMOKE_BATCH_SIZE
        ),
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "model_source": file_record(
        MODEL_SOURCE
    ),
    "model_registry": file_record(
        MODEL_REGISTRY
    ),
    "scaler": file_record(
        SCALER_NPZ
    ),
    "class_weights": file_record(
        CLASS_WEIGHTS_NPZ
    ),
    "architectures": (
        architecture_reports
    ),
    "checks": all_checks,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

completion = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "neural_architecture_one_batch_smoke_test"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "architecture_count": 2,
    "architectures": [
        {
            "architecture": row[
                "architecture"
            ],
            "selected_symbol": row[
                "selected_symbol"
            ],
            "linear_widths": row[
                "linear_widths"
            ],
            "parameter_count": row[
                "parameter_count"
            ],
            "linear_macs_per_sample": row[
                "linear_macs_per_sample"
            ],
        }
        for row in architecture_reports
    ],
    "validation_access_count": 0,
    "test_access_count": 0,
    "report": str(
        OUTPUT_REPORT
    ),
    "report_sha256": (
        sha256_file(
            OUTPUT_REPORT
        )
    ),
    "ready_for_B0_training_runs": True,
    "next_action": (
        "Run and verify the ten locked B0 "
        "training configurations."
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

post_checks = {
    "report_exists": (
        OUTPUT_REPORT.exists()
    ),
    "completion_exists": (
        OUTPUT_COMPLETION.exists()
    ),
    "completion_report_hash_matches": (
        read_json(
            OUTPUT_COMPLETION
        )["report_sha256"]
        == sha256_file(
            OUTPUT_REPORT
        )
    ),
    "completion_ready_for_B0": (
        read_json(
            OUTPUT_COMPLETION
        ).get(
            "ready_for_B0_training_runs"
        )
        is True
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Phase 5 smoke-test post-lock "
        "checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 NEURAL SMOKE TEST SUMMARY")
print("=" * 92)

for row in architecture_reports:
    print(
        f"{row['architecture']:<16} | "
        f"symbol={row['selected_symbol']:<24} | "
        f"parameters={row['parameter_count']:,} | "
        f"MACs/sample={row['linear_macs_per_sample']:,}"
    )

print(
    "Forward pass                    : PASSED"
)
print(
    "Weighted loss                   : PASSED"
)
print(
    "Backward pass                   : PASSED"
)
print(
    "Optimizer update                : PASSED"
)
print(
    "Deterministic initialization    : PASSED"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "Ready for ten B0 training runs  : True"
)
print(
    "Smoke-test report               : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 NEURAL ARCHITECTURE SMOKE TEST LOCKED"
)
