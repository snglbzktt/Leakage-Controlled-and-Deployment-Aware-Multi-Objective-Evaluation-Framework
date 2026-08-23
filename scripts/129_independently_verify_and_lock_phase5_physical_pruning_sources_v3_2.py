from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PRUNING_ENGINE_LOCK = (
    AUDIT
    / "phase5_physical_pruning_engine_locked_v3_2.json"
)

PRUNING_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

PRUNING_SOURCES_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_locked_v3_2.json"
)

PRUNING_SOURCES_SUMMARY = (
    AUDIT
    / "phase5_physical_pruning_sources_summary_v3_2.json"
)

PRUNING_SOURCES_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

PRUNING_SOURCES_GENERATION_MANIFEST = (
    AUDIT
    / "phase5_physical_pruning_sources_lock_manifest_v3_2.json"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"

SCALER_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_physical_pruning_sources_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

VARIANT_TO_RATIO = {
    "P25-noFT": 0.25,
    "P50-noFT": 0.50,
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        "P25-noFT": [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        "P25-noFT": [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
}

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
SMOKE_BATCH_SIZE = 97
CPU_THREADS = 4


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


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
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


def load_module(
    path: Path,
    module_name: str,
) -> ModuleType:
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
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

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


def callable_kwargs(
    candidate: Any,
    architecture: str,
) -> dict[str, Any]:
    signature = inspect.signature(
        candidate
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

        normalized = "".join(
            character
            for character in name.lower()
            if character.isalnum()
        )

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
                EXPECTED_INPUT_FEATURES
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
                EXPECTED_OUTPUT_CLASSES
            )
            continue

        if normalized in {
            "hiddendims",
            "hiddensizes",
            "hiddenlayers",
            "hiddenunits",
        }:
            kwargs[name] = list(
                HIDDEN_DIMS[
                    architecture
                ]
            )
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Unresolved model constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    model_symbol: str,
    architecture: str,
) -> nn.Module:
    if not hasattr(
        model_module,
        model_symbol,
    ):
        raise RuntimeError(
            f"Missing model symbol: {model_symbol}"
        )

    candidate = getattr(
        model_module,
        model_symbol,
    )

    model = candidate(
        **callable_kwargs(
            candidate,
            architecture,
        )
    )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Model constructor did not "
            "return torch.nn.Module."
        )

    return model


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


def state_dict_digest(
    state_dict: dict[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(
        state_dict.items()
    ):
        contiguous = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )

        digest.update(
            str(
                contiguous.dtype
            ).encode("ascii")
        )

        digest.update(
            np.asarray(
                contiguous.shape,
                dtype=np.int64,
            ).tobytes()
        )

        digest.update(
            contiguous.numpy().tobytes()
        )

    return digest.hexdigest()


def state_dicts_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> bool:
    if set(
        first.keys()
    ) != set(
        second.keys()
    ):
        return False

    return all(
        torch.equal(
            first[key],
            second[key],
        )
        for key in first
    )


required_paths = (
    PHASE5_PROTOCOL,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE_PATH,
    PRUNING_SOURCES_LOCK,
    PRUNING_SOURCES_SUMMARY,
    PRUNING_SOURCES_REGISTRY,
    PRUNING_SOURCES_GENERATION_MANIFEST,
    MODEL_SOURCE,
    X_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Independent pruning-source "
            "verification artifact already "
            f"exists: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

engine_lock = read_json(
    PRUNING_ENGINE_LOCK
)

sources_lock = read_json(
    PRUNING_SOURCES_LOCK
)

sources_summary = read_json(
    PRUNING_SOURCES_SUMMARY
)

generation_manifest = read_json(
    PRUNING_SOURCES_GENERATION_MANIFEST
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

source_rows = read_csv(
    PRUNING_SOURCES_REGISTRY
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "B0_pair_locked": (
        B0_pair_lock.get("status")
        == "locked"
        and B0_pair_lock.get(
            "ready_for_branch_source_generation"
        )
        is True
    ),
    "B0_registry_hash_matches": (
        B0_pair_lock.get(
            "checkpoint_registry_sha256"
        )
        == sha256_file(
            B0_CHECKPOINT_REGISTRY
        )
    ),
    "pruning_engine_locked": (
        engine_lock.get("status")
        == "locked"
        and engine_lock.get(
            "ready_for_P25_noFT_and_P50_noFT_generation"
        )
        is True
    ),
    "pruning_engine_hash_matches": (
        engine_lock.get(
            "engine_source_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_PATH
        )
    ),
    "source_generation_locked": (
        sources_lock.get("status")
        == "locked"
        and sources_lock.get(
            "ready_for_independent_pruning_source_verification"
        )
        is True
        and sources_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "source_registry_hash_matches": (
        sources_lock.get(
            "registry_sha256"
        )
        == sha256_file(
            PRUNING_SOURCES_REGISTRY
        )
    ),
    "source_summary_hash_matches": (
        sources_lock.get(
            "summary_sha256"
        )
        == sha256_file(
            PRUNING_SOURCES_SUMMARY
        )
    ),
    "generation_manifest_locked": (
        generation_manifest.get(
            "status"
        )
        == "locked"
        and generation_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "source_summary_completed": (
        sources_summary.get(
            "status"
        )
        == "completed"
        and sources_summary.get(
            "checkpoint_count"
        )
        == 20
        and sources_summary.get(
            "all_checks_passed"
        )
        is True
    ),
    "B0_registry_has_ten_rows": (
        len(B0_rows)
        == 10
    ),
    "source_registry_has_twenty_rows": (
        len(source_rows)
        == 20
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
        "Independent pruning-source "
        "verification entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

torch.set_num_threads(
    CPU_THREADS
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.use_deterministic_algorithms(
    True
)

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

with np.load(
    SCALER_NPZ
) as values:
    mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

smoke_numpy = (
    (
        np.asarray(
            x_train[
                :SMOKE_BATCH_SIZE
            ],
            dtype=np.float64,
        )
        - mean64
    )
    / scale64
).astype(np.float32)

if not np.isfinite(
    smoke_numpy
).all():
    raise RuntimeError(
        "Non-finite train-only verification batch."
    )

smoke_batch = torch.from_numpy(
    smoke_numpy
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_pruning_sources_verify_models",
)

engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase5_pruning_sources_verify_engine",
)

if (
    engine_module.ENGINE_VERSION
    != ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
}

expected_source_keys = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANT_TO_RATIO
    for seed in SEEDS
}

observed_source_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in source_rows
}

if (
    observed_source_keys
    != expected_source_keys
):
    raise RuntimeError(
        "Pruning-source architecture, "
        "variant, and seed matrix mismatch."
    )

verified_rows: list[
    dict[str, Any]
] = []

for index, source_row in enumerate(
    sorted(
        source_rows,
        key=lambda row: (
            row["architecture"],
            row["variant"],
            int(row["seed"]),
        ),
    ),
    start=1,
):
    architecture = (
        source_row["architecture"]
    )

    variant = source_row["variant"]

    seed = int(
        source_row["seed"]
    )

    pruning_ratio = (
        VARIANT_TO_RATIO[
            variant
        ]
    )

    B0_row = B0_lookup[
        (
            architecture,
            seed,
        )
    ]

    B0_checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    source_checkpoint_path = Path(
        source_row[
            "source_checkpoint_path"
        ]
    )

    source_manifest_path = Path(
        source_row[
            "source_manifest_path"
        ]
    )

    for path in (
        B0_checkpoint_path,
        source_checkpoint_path,
        source_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_manifest = read_json(
        source_manifest_path
    )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    stored_source = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        source_row["model_symbol"]
    )

    B0_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    B0_model.load_state_dict(
        B0_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    B0_model.eval()

    (
        regenerated_model,
        regenerated_metadata,
    ) = engine_module.physically_prune_mlp(
        B0_model,
        pruning_ratio,
        source_checkpoint_sha256=(
            B0_row[
                "checkpoint_sha256"
            ]
        ),
    )

    regenerated_model.eval()

    stored_state_dict = stored_source[
        "model_state_dict"
    ]

    regenerated_state_dict = (
        regenerated_model.state_dict()
    )

    reloaded_model, _ = (
        engine_module.physically_prune_mlp(
            B0_model,
            pruning_ratio,
            source_checkpoint_sha256=(
                B0_row[
                    "checkpoint_sha256"
                ]
            ),
        )
    )

    reloaded_model.load_state_dict(
        stored_state_dict,
        strict=True,
    )

    reloaded_model.eval()

    with torch.inference_mode():
        regenerated_logits = (
            extract_logits(
                regenerated_model(
                    smoke_batch
                )
            )
        )

        reloaded_logits = (
            extract_logits(
                reloaded_model(
                    smoke_batch
                )
            )
        )

    expected_widths = (
        EXPECTED_WIDTHS[
            architecture
        ][variant]
    )

    checks = {
        "source_checkpoint_hash_matches_registry": (
            sha256_file(
                source_checkpoint_path
            )
            == source_row[
                "source_checkpoint_sha256"
            ]
        ),
        "source_manifest_hash_matches_registry": (
            sha256_file(
                source_manifest_path
            )
            == source_row[
                "source_manifest_sha256"
            ]
        ),
        "B0_checkpoint_hash_matches_registry": (
            sha256_file(
                B0_checkpoint_path
            )
            == B0_row[
                "checkpoint_sha256"
            ]
            == source_row[
                "source_B0_checkpoint_sha256"
            ]
        ),
        "manifest_identity_matches": (
            source_manifest.get(
                "status"
            )
            == "completed"
            and source_manifest.get(
                "architecture"
            )
            == architecture
            and source_manifest.get(
                "variant"
            )
            == variant
            and int(
                source_manifest.get(
                    "seed"
                )
            )
            == seed
        ),
        "stored_checkpoint_identity_matches": (
            stored_source.get(
                "artifact_type"
            )
            == (
                "phase5_physical_pruning_source"
            )
            and stored_source.get(
                "architecture"
            )
            == architecture
            and stored_source.get(
                "variant"
            )
            == variant
            and int(
                stored_source.get(
                    "seed"
                )
            )
            == seed
        ),
        "engine_version_matches": (
            stored_source.get(
                "engine_version"
            )
            == ENGINE_VERSION
            and stored_source.get(
                "pruning_engine_sha256"
            )
            == sha256_file(
                PRUNING_ENGINE_PATH
            )
        ),
        "stored_metadata_matches_manifest": (
            stored_source[
                "pruning_metadata"
            ][
                "compacted_linear_widths"
            ]
            == source_manifest[
                "compacted_linear_widths"
            ]
            and stored_source[
                "pruning_metadata"
            ][
                "compacted_parameter_count"
            ]
            == source_manifest[
                "compacted_parameter_count"
            ]
            and stored_source[
                "pruning_metadata"
            ][
                "compacted_linear_macs_per_sample"
            ]
            == source_manifest[
                "compacted_linear_macs_per_sample"
            ]
        ),
        "expected_widths_match": (
            regenerated_metadata[
                "compacted_linear_widths"
            ]
            == expected_widths
            and stored_source[
                "pruning_metadata"
            ][
                "compacted_linear_widths"
            ]
            == expected_widths
            and engine_module.inspect_linear_widths(
                reloaded_model
            )
            == expected_widths
        ),
        "regenerated_metadata_matches_stored": (
            regenerated_metadata
            == stored_source[
                "pruning_metadata"
            ]
        ),
        "regenerated_state_matches_stored_exactly": (
            state_dicts_equal(
                regenerated_state_dict,
                stored_state_dict,
            )
        ),
        "regenerated_state_digest_matches": (
            state_dict_digest(
                regenerated_state_dict
            )
            == state_dict_digest(
                stored_state_dict
            )
        ),
        "reloaded_output_matches_regenerated": (
            torch.equal(
                regenerated_logits,
                reloaded_logits,
            )
        ),
        "output_shape_matches": (
            tuple(
                reloaded_logits.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_OUTPUT_CLASSES,
            )
        ),
        "output_is_finite": (
            bool(
                torch.isfinite(
                    reloaded_logits
                ).all().item()
            )
        ),
        "physical_compaction_true": (
            stored_source[
                "pruning_metadata"
            ][
                "physical_compaction"
            ]
            is True
            and stored_source[
                "pruning_metadata"
            ][
                "mask_only_pruning"
            ]
            is False
        ),
        "generated_directly_from_B0": (
            stored_source[
                "pruning_metadata"
            ][
                "generated_directly_from_B0"
            ]
            is True
        ),
        "parameter_count_reduced": (
            stored_source[
                "pruning_metadata"
            ][
                "compacted_parameter_count"
            ]
            < stored_source[
                "pruning_metadata"
            ][
                "source_parameter_count"
            ]
        ),
        "MAC_count_reduced": (
            stored_source[
                "pruning_metadata"
            ][
                "compacted_linear_macs_per_sample"
            ]
            < stored_source[
                "pruning_metadata"
            ][
                "source_linear_macs_per_sample"
            ]
        ),
        "validation_access_count_zero": (
            int(
                stored_source.get(
                    "validation_access_count"
                )
            )
            == 0
            and int(
                source_manifest.get(
                    "validation_access_count"
                )
            )
            == 0
            and int(
                source_row[
                    "validation_access_count"
                ]
            )
            == 0
        ),
        "test_access_count_zero": (
            int(
                stored_source.get(
                    "test_access_count"
                )
            )
            == 0
            and int(
                source_manifest.get(
                    "test_access_count"
                )
            )
            == 0
            and int(
                source_row[
                    "test_access_count"
                ]
            )
            == 0
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
            f"{architecture} seed {seed} "
            f"{variant} independent "
            "verification failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_rows.append(
        {
            "architecture": (
                architecture
            ),
            "variant": variant,
            "seed": seed,
            "pruning_ratio": (
                pruning_ratio
            ),
            "source_checkpoint": (
                file_record(
                    source_checkpoint_path
                )
            ),
            "source_manifest": (
                file_record(
                    source_manifest_path
                )
            ),
            "source_B0_checkpoint": (
                file_record(
                    B0_checkpoint_path
                )
            ),
            "compacted_linear_widths": (
                expected_widths
            ),
            "compacted_parameter_count": (
                stored_source[
                    "pruning_metadata"
                ][
                    "compacted_parameter_count"
                ]
            ),
            "compacted_linear_macs_per_sample": (
                stored_source[
                    "pruning_metadata"
                ][
                    "compacted_linear_macs_per_sample"
                ]
            ),
            "state_dict_sha256": (
                state_dict_digest(
                    stored_state_dict
                )
            ),
            "validation_access_count": 0,
            "test_access_count": 0,
            "checks": checks,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/20] "
        f"{architecture} seed={seed} "
        f"{variant} | "
        "regenerated_state=exact | "
        "output=exact | "
        "validation=0 | test=0 | "
        "all checks=True",
        flush=True,
    )

    del B0_model
    del regenerated_model
    del reloaded_model
    del B0_checkpoint
    del stored_source

global_checks = {
    "twenty_sources_verified": (
        len(
            verified_rows
        )
        == 20
    ),
    "complete_architecture_variant_seed_matrix": (
        {
            (
                row["architecture"],
                row["variant"],
                int(row["seed"]),
            )
            for row in verified_rows
        }
        == expected_source_keys
    ),
    "all_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_rows
        )
    ),
    "all_state_dict_hashes_unique": (
        len(
            {
                row[
                    "state_dict_sha256"
                ]
                for row in verified_rows
            }
        )
        == 20
    ),
    "all_validation_counts_zero": (
        all(
            row[
                "validation_access_count"
            ]
            == 0
            for row in verified_rows
        )
    ),
    "all_test_counts_zero": (
        all(
            row[
                "test_access_count"
            ]
            == 0
            for row in verified_rows
        )
    ),
}

failed_global_checks = [
    name
    for name, passed
    in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "Independent pruning-source "
        "global verification failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "checkpoint_count": 20,
    "architecture_count": 2,
    "variant_count": 2,
    "seeds": list(SEEDS),
    "verification_policy": {
        "source_checkpoint_hashes_verified": True,
        "source_manifest_hashes_verified": True,
        "B0_checkpoint_hashes_verified": True,
        "physical_pruning_regenerated_from_B0": True,
        "regenerated_state_compared_exactly": True,
        "train_only_smoke_output_compared_exactly": True,
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "verified_sources": (
        verified_rows
    ),
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
        "phase5_protocol": (
            file_record(
                PHASE5_PROTOCOL
            )
        ),
        "B0_pair_lock": (
            file_record(
                B0_PAIR_LOCK
            )
        ),
        "B0_checkpoint_registry": (
            file_record(
                B0_CHECKPOINT_REGISTRY
            )
        ),
        "pruning_engine_lock": (
            file_record(
                PRUNING_ENGINE_LOCK
            )
        ),
        "pruning_engine_source": (
            file_record(
                PRUNING_ENGINE_PATH
            )
        ),
        "pruning_sources_lock": (
            file_record(
                PRUNING_SOURCES_LOCK
            )
        ),
        "pruning_sources_summary": (
            file_record(
                PRUNING_SOURCES_SUMMARY
            )
        ),
        "pruning_sources_registry": (
            file_record(
                PRUNING_SOURCES_REGISTRY
            )
        ),
    },
    "ready_for_P25_noFT_and_P50_noFT_evaluation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints_"
        "independently_verified"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "checkpoint_count": 20,
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "source_registry": str(
        PRUNING_SOURCES_REGISTRY
    ),
    "source_registry_sha256": (
        sha256_file(
            PRUNING_SOURCES_REGISTRY
        )
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_P25_noFT_and_P50_noFT_evaluation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints_"
        "independently_verified"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            B0_PAIR_LOCK
        ),
        file_record(
            B0_CHECKPOINT_REGISTRY
        ),
        file_record(
            PRUNING_ENGINE_LOCK
        ),
        file_record(
            PRUNING_ENGINE_PATH
        ),
        file_record(
            PRUNING_SOURCES_LOCK
        ),
        file_record(
            PRUNING_SOURCES_SUMMARY
        ),
        file_record(
            PRUNING_SOURCES_REGISTRY
        ),
        file_record(
            PRUNING_SOURCES_GENERATION_MANIFEST
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            SCALER_NPZ
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "checkpoint_count": 20,
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_P25_noFT_and_P50_noFT_evaluation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "verification_exists": (
        OUTPUT_VERIFICATION.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "verification_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "verification_report_sha256"
        ]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "registry_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "source_registry_sha256"
        ]
        == sha256_file(
            PRUNING_SOURCES_REGISTRY
        )
    ),
    "ready_for_noFT_evaluation": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_P25_noFT_and_P50_noFT_evaluation"
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
        "Independent pruning-source "
        "post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 PHYSICAL PRUNING SOURCE INDEPENDENT VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Source checkpoints verified     : 20"
)
print(
    "B0 regeneration performed       : 20"
)
print(
    "Exact state comparisons         : PASSED"
)
print(
    "Exact smoke-output comparisons  : PASSED"
)
print(
    "Checkpoint and manifest hashes  : PASSED"
)
print(
    "Physical compaction verified    : PASSED"
)
print(
    "Direct B0 derivation verified   : PASSED"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "Independent verification status : LOCKED"
)
print(
    "Ready for P25/P50 noFT evaluate : True"
)
print(
    "Verification report             : "
    f"{OUTPUT_VERIFICATION}"
)
print(
    "Verified lock                   : "
    f"{OUTPUT_LOCK}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 PHYSICAL PRUNING SOURCES INDEPENDENTLY VERIFIED AND LOCKED"
)
