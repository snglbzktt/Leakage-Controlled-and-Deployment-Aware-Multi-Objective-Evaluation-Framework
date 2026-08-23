from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
CACHE = ROOT / "data" / "cache" / "tabular_baseline_family3_v3_2"

PROTOCOL = ROOT / "configs" / "protocols" / "phase5_fair_budget_compression_protocol_v3_2.json"
MATRIX = AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
B0_PAIR_LOCK = AUDIT / "phase5_B0_pair_locked_v3_2.json"
B0_REGISTRY = AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
PRUNING_ENGINE_LOCK = AUDIT / "phase5_physical_pruning_engine_locked_v3_2.json"
PRUNING_ENGINE = ROOT / "src" / "compression" / "phase5_physical_pruning_engine_v3_2.py"
PRUNING_SOURCES_LOCK = AUDIT / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
PRUNING_REGISTRY = AUDIT / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
QAT_PREFLIGHT_LOCK = AUDIT / "phase5_QAT_runtime_preflight_locked_v3_2.json"
QAT_PREFLIGHT_MATRIX = AUDIT / "phase5_QAT_runtime_preflight_matrix_v3_2.csv"
MODEL_SOURCE = ROOT / "src" / "models" / "nbaiot_models.py"
QAT_ENGINE = ROOT / "src" / "compression" / "phase5_qat_engine_v3_2.py"
SCALER = ROOT / "results" / "v2" / "phase5_compression" / "shared" / "preprocessing" / "phase5_train_only_standard_scaler_v3_2.npz"

RUNS_CSV = AUDIT / "phase5_QAT_all_runs_v3_2.csv"
GROUPS_CSV = AUDIT / "phase5_QAT_group_summary_v3_2.csv"
SUMMARY_JSON = AUDIT / "phase5_QAT_summary_v3_2.json"
COMPLETION_JSON = AUDIT / "phase5_QAT_completed_v3_2.json"

OUT_RUNS = AUDIT / "phase5_QAT_verified_runs_v3_2.csv"
OUT_VERIFY = AUDIT / "phase5_QAT_verification_v3_2.json"
OUT_LOCK = AUDIT / "phase5_QAT_locked_v3_2.json"
OUT_MANIFEST = AUDIT / "phase5_QAT_lock_manifest_v3_2.json"

X_VAL = CACHE / "X_validation.npy"
Y_VAL = CACHE / "y_validation.npy"
RAW_VAL = CACHE / "raw_row_count_validation.npy"
Y_TEST = CACHE / "y_test.npy"
RAW_TEST = CACHE / "raw_row_count_test.npy"

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
QAT_ENGINE_VERSION = "phase5_qat_engine_v3_2"
PRUNING_ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

ARCHS = ("tinyml_mlp", "compact_dnn")
VARIANTS = ("QAT", "P25-QAT", "P50-QAT")
SEEDS = (42, 123, 2026, 3407, 8192)
EXPECTED_KEYS = {(a, v, s) for a in ARCHS for v in VARIANTS for s in SEEDS}

SOURCE_VARIANT = {
    "QAT": "B0",
    "P25-QAT": "P25-noFT",
    "P50-QAT": "P50-noFT",
}
PRUNE_RATIO = {"P25-QAT": 0.25, "P50-QAT": 0.50}
SUBDIR = {"QAT": "qat", "P25-QAT": "p25_qat", "P50-QAT": "p50_qat"}
BASE_LR = {"tinyml_mlp": 0.003, "compact_dnn": 0.001}
QAT_LR = {k: v * 0.1 for k, v in BASE_LR.items()}
HIDDEN = {"tinyml_mlp": [64, 32], "compact_dnn": [128, 64, 32]}
WIDTHS = {
    "tinyml_mlp": {
        "QAT": [[115, 64], [64, 32], [32, 3]],
        "P25-QAT": [[115, 48], [48, 24], [24, 3]],
        "P50-QAT": [[115, 32], [32, 16], [16, 3]],
    },
    "compact_dnn": {
        "QAT": [[115, 128], [128, 64], [64, 32], [32, 3]],
        "P25-QAT": [[115, 96], [96, 48], [48, 24], [24, 3]],
        "P50-QAT": [[115, 64], [64, 32], [32, 16], [16, 3]],
    },
}

LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
VAL_ROWS = 371_797
TEST_ROWS = 371_796
VAL_RAW_ROWS = 1_059_390
TEST_RAW_ROWS = 1_059_393
FEATURES = 115
CLASSES = 3
BATCH = 4096
MAX_EPOCHS = 8
PATIENCE = 3
MIN_DELTA = 0.0002
WEIGHT_DECAY = 0.0001
EVAL_BATCH = 16_384
TOL = 1e-10

warnings.filterwarnings("ignore", message="torch.ao.quantization is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Please use quant_min and quant_max.*", category=UserWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", category=UserWarning)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha(path)}


def close(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOL)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(value)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ctor_kwargs(candidate: Any, architecture: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in inspect.signature(candidate).parameters.items():
        if name in {"self", "args", "kwargs"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        key = "".join(c for c in name.lower() if c.isalnum())
        if key in {"inputdim", "inputsize", "inputfeatures", "nfeatures", "numfeatures", "infeatures", "featurecount"}:
            result[name] = FEATURES
        elif key in {"numclasses", "nclasses", "classcount", "outputdim", "outputsize", "outfeatures"}:
            result[name] = CLASSES
        elif key in {"hiddendims", "hiddensizes", "hiddenlayers", "hiddenunits"}:
            result[name] = list(HIDDEN[architecture])
        elif parameter.default is inspect.Parameter.empty:
            raise RuntimeError(f"Unresolved constructor parameter: {name}")
    return result


def instantiate(module: ModuleType, symbol: str, architecture: str) -> nn.Module:
    candidate = getattr(module, symbol)
    model = candidate(**ctor_kwargs(candidate, architecture))
    if not isinstance(model, nn.Module):
        raise RuntimeError("Constructor did not return nn.Module.")
    return model


def run_dir(architecture: str, variant: str, seed: int) -> Path:
    return ROOT / "results" / "v2" / "phase5_compression" / architecture / SUBDIR[variant] / f"seed_{seed}"


def build_source(
    architecture: str,
    variant: str,
    seed: int,
    model_module: ModuleType,
    pruning_module: ModuleType,
    b0_lookup: dict[tuple[str, int], dict[str, str]],
    pruning_lookup: dict[tuple[str, str, int], dict[str, str]],
) -> tuple[nn.Module, str, Path, str, str, dict[str, Any] | None]:
    if variant == "QAT":
        row = b0_lookup[(architecture, seed)]
        path = Path(row["checkpoint_path"])
        digest = sha(path)
        if digest != row["checkpoint_sha256"]:
            raise RuntimeError("B0 hash mismatch.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        symbol = row["model_symbol"]
        model = instantiate(model_module, symbol, architecture)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, symbol, path, digest, "B0", None

    source_variant = SOURCE_VARIANT[variant]
    row = pruning_lookup[(architecture, source_variant, seed)]
    source_path = Path(row["source_checkpoint_path"])
    source_digest = sha(source_path)
    if source_digest != row["source_checkpoint_sha256"]:
        raise RuntimeError("Pruning-source hash mismatch.")

    b0_path = Path(row["source_B0_checkpoint_path"])
    b0_digest = sha(b0_path)
    if b0_digest != row["source_B0_checkpoint_sha256"]:
        raise RuntimeError("Pruning B0 hash mismatch.")

    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    b0_checkpoint = torch.load(b0_path, map_location="cpu", weights_only=False)
    symbol = row["model_symbol"]

    b0_model = instantiate(model_module, symbol, architecture)
    b0_model.load_state_dict(b0_checkpoint["model_state_dict"], strict=True)

    model, metadata = pruning_module.physically_prune_mlp(
        b0_model,
        PRUNE_RATIO[variant],
        source_checkpoint_sha256=b0_digest,
    )
    if metadata != source_checkpoint["pruning_metadata"]:
        raise RuntimeError("Pruning metadata mismatch.")
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, symbol, source_path, source_digest, source_variant, metadata


def transform(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = ((np.asarray(x, dtype=np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite scaled value.")
    return z


def infer_validation(
    model: nn.Module,
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    qat_module: ModuleType,
) -> tuple[np.ndarray, np.ndarray, float]:
    pred = np.empty(len(x), dtype=np.int8)
    prob = np.empty((len(x), CLASSES), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(x), EVAL_BATCH):
            end = min(start + EVAL_BATCH, len(x))
            xb = torch.from_numpy(transform(x[start:end], mean, scale))
            logits = qat_module.extract_logits(model(xb))
            p = torch.softmax(logits, dim=1)
            prob[start:end] = p.cpu().numpy().astype(np.float32)
            pred[start:end] = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int8)
    return pred, prob, time.perf_counter() - started


def metric_view(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        sample_weight=sample_weight,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
        sample_weight=sample_weight,
    )
    matrix = matrix.astype(np.int64) if sample_weight is None else np.rint(matrix).astype(np.int64)
    support_total = float(np.sum(support))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support) if support_total > 0 else 0.0),
        "mcc": float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y_true, probabilities, labels=LABELS, sample_weight=sample_weight)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=LABELS,
                multi_class="ovr",
                average="macro",
                sample_weight=sample_weight,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": {},
    }
    for i, name in enumerate(CLASS_NAMES):
        result["per_class"][name] = {
            "label": int(LABELS[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
            "false_negative_rate": float(1.0 - recall[i]),
        }
    return result


def metrics_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    for key in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "roc_auc_ovr_macro",
    ):
        if not close(a[key], b[key]):
            return False
    if a["confusion_matrix"] != b["confusion_matrix"]:
        return False
    for name in CLASS_NAMES:
        for key in ("label", "precision", "recall", "f1", "support", "false_negative_rate"):
            if key == "label":
                if int(a["per_class"][name][key]) != int(b["per_class"][name][key]):
                    return False
            elif not close(a["per_class"][name][key], b["per_class"][name][key]):
                return False
    return True


def history_ok(rows: list[dict[str, str]], best_epoch_expected: int, best_value_expected: float) -> bool:
    best = -math.inf
    best_epoch = 0
    wait = 0
    for row in rows:
        epoch = int(row["epoch"])
        value = float(row["validation_INT8_macro_f1"])
        improved = best_epoch == 0 or value > best + MIN_DELTA
        if parse_bool(row["improved"]) != improved:
            return False
        if improved:
            best = value
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if int(row["epochs_without_improvement"]) != wait:
            return False
    return best_epoch == best_epoch_expected and close(best, best_value_expected)


def manifest_artifacts_ok(directory: Path, manifest: dict[str, Any]) -> bool:
    expected = {str(r["relative_path"]) for r in manifest["artifacts"]}
    actual = {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name not in {"run_manifest.json", "run_status.json"}
    }
    if expected != actual:
        return False
    for item in manifest["artifacts"]:
        path = directory / str(item["relative_path"])
        if not path.exists():
            return False
        if int(item["size_bytes"]) != int(path.stat().st_size):
            return False
        if item["sha256"] != sha(path):
            return False
    return True


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std_population": float(np.std(x, ddof=0)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


required = (
    X_VAL,
    Y_VAL,
    RAW_VAL,
    Y_TEST,
    RAW_TEST,
    PROTOCOL,
    MATRIX,
    B0_PAIR_LOCK,
    B0_REGISTRY,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE,
    PRUNING_SOURCES_LOCK,
    PRUNING_REGISTRY,
    QAT_PREFLIGHT_LOCK,
    QAT_PREFLIGHT_MATRIX,
    MODEL_SOURCE,
    QAT_ENGINE,
    SCALER,
    RUNS_CSV,
    GROUPS_CSV,
    SUMMARY_JSON,
    COMPLETION_JSON,
)
for path in required:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (OUT_RUNS, OUT_VERIFY, OUT_LOCK, OUT_MANIFEST):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")

protocol = read_json(PROTOCOL)
b0_pair_lock = read_json(B0_PAIR_LOCK)
pruning_engine_lock = read_json(PRUNING_ENGINE_LOCK)
pruning_sources_lock = read_json(PRUNING_SOURCES_LOCK)
qat_preflight_lock = read_json(QAT_PREFLIGHT_LOCK)
summary = read_json(SUMMARY_JSON)
completion = read_json(COMPLETION_JSON)

runs_rows = read_csv(RUNS_CSV)
groups_rows = read_csv(GROUPS_CSV)
matrix_rows = read_csv(MATRIX)
b0_rows = read_csv(B0_REGISTRY)
pruning_rows = read_csv(PRUNING_REGISTRY)
preflight_rows = read_csv(QAT_PREFLIGHT_MATRIX)

run_lookup = {(r["architecture"], r["variant"], int(r["seed"])): r for r in runs_rows}
matrix_lookup = {
    (r["architecture"], r["variant"], int(r["seed"])): r
    for r in matrix_rows
    if r["architecture"] in ARCHS and r["variant"] in VARIANTS
}
b0_lookup = {(r["architecture"], int(r["seed"])): r for r in b0_rows}
pruning_lookup = {(r["architecture"], r["variant"], int(r["seed"])): r for r in pruning_rows}
preflight_lookup = {(r["architecture"], r["variant"], int(r["seed"])): r for r in preflight_rows}

entry_checks = {
    "protocol_locked": protocol.get("status") == "locked" and protocol.get("protocol_version") == PROTOCOL_VERSION,
    "B0_pair_locked": b0_pair_lock.get("status") == "locked" and b0_pair_lock.get("all_checks_passed") is True,
    "pruning_engine_locked": pruning_engine_lock.get("status") == "locked" and pruning_engine_lock.get("all_checks_passed") is True,
    "pruning_engine_hash": pruning_engine_lock.get("engine_source_sha256") == sha(PRUNING_ENGINE),
    "pruning_sources_locked": pruning_sources_lock.get("status") == "locked" and pruning_sources_lock.get("all_checks_passed") is True,
    "QAT_preflight_locked": qat_preflight_lock.get("status") == "locked" and qat_preflight_lock.get("all_checks_passed") is True,
    "QAT_engine_hash": qat_preflight_lock.get("QAT_engine_source_sha256") == sha(QAT_ENGINE),
    "QAT_preflight_matrix_hash": qat_preflight_lock.get("matrix_csv_sha256") == sha(QAT_PREFLIGHT_MATRIX),
    "summary_complete": summary.get("status") == "completed" and summary.get("run_count") == 30,
    "completion_complete": completion.get("status") == "completed" and completion.get("run_count") == 30,
    "aggregate_hashes": (
        completion.get("aggregate_json_sha256") == sha(SUMMARY_JSON)
        and completion.get("aggregate_runs_csv_sha256") == sha(RUNS_CSV)
        and completion.get("group_summary_csv_sha256") == sha(GROUPS_CSV)
    ),
    "run_matrix_complete": len(runs_rows) == 30 and set(run_lookup) == EXPECTED_KEYS,
    "group_matrix_complete": len(groups_rows) == 6,
    "locked_matrix_complete": set(matrix_lookup) == EXPECTED_KEYS,
    "preflight_matrix_complete": set(preflight_lookup) == EXPECTED_KEYS,
}
failed = [k for k, v in entry_checks.items() if not v]
if failed:
    raise RuntimeError("QAT verification entry gate failed: " + ", ".join(failed))

x_val = np.load(X_VAL, mmap_mode="r")
y_val = np.asarray(np.load(Y_VAL, mmap_mode="r"), dtype=np.int64)
raw_val = np.asarray(np.load(RAW_VAL, mmap_mode="r"), dtype=np.int64)
y_test = np.asarray(np.load(Y_TEST, mmap_mode="r"), dtype=np.int64)
raw_test = np.asarray(np.load(RAW_TEST, mmap_mode="r"), dtype=np.int64)

with np.load(SCALER) as values:
    mean64 = np.asarray(values["mean_float64"], dtype=np.float64)
    scale64 = np.asarray(values["scale_float64"], dtype=np.float64)

cache_checks = {
    "validation_shapes": x_val.shape == (VAL_ROWS, FEATURES) and y_val.shape == (VAL_ROWS,) and raw_val.shape == (VAL_ROWS,),
    "test_shapes": y_test.shape == (TEST_ROWS,) and raw_test.shape == (TEST_ROWS,),
    "raw_totals": int(raw_val.sum()) == VAL_RAW_ROWS and int(raw_test.sum()) == TEST_RAW_ROWS,
    "scaler_shapes": mean64.shape == (FEATURES,) and scale64.shape == (FEATURES,),
}
failed = [k for k, v in cache_checks.items() if not v]
if failed:
    raise RuntimeError("QAT verification cache check failed: " + ", ".join(failed))

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True)

backend = str(qat_preflight_lock["selected_backend"])
if backend not in torch.backends.quantized.supported_engines:
    raise RuntimeError("Locked QAT backend unavailable.")
torch.backends.quantized.engine = backend

model_module = load_module(MODEL_SOURCE, "phase5_qat_verify_models")
pruning_module = load_module(PRUNING_ENGINE, "phase5_qat_verify_pruning")
qat_module = load_module(QAT_ENGINE, "phase5_qat_verify_engine")

if pruning_module.ENGINE_VERSION != PRUNING_ENGINE_VERSION:
    raise RuntimeError("Pruning engine version mismatch.")
if qat_module.ENGINE_VERSION != QAT_ENGINE_VERSION:
    raise RuntimeError("QAT engine version mismatch.")

print("=" * 92)
print("PHASE 5 QAT INDEPENDENT VERIFICATION")
print("=" * 92)
print("Runs                            : 30")
print("QAT checkpoint loading          : Yes")
print("INT8 checkpoint loading         : Yes")
print("Optimizer-state loading         : Yes")
print("Independent validation inference: Yes")
print("Test model inference            : No")
print("Test verification source        : Saved predictions only")
print(f"QAT backend                     : {backend}")
print()

verified: list[dict[str, Any]] = []

for index, key in enumerate(sorted(EXPECTED_KEYS), start=1):
    architecture, variant, seed = key
    aggregate_row = run_lookup[key]
    matrix_row = matrix_lookup[key]
    preflight_row = preflight_lookup[key]
    directory = run_dir(architecture, variant, seed)

    paths = {
        "status": directory / "run_status.json",
        "manifest": directory / "run_manifest.json",
        "metrics": directory / "metrics.json",
        "history": directory / "training_history.csv",
        "qat": directory / "best_qat_checkpoint.pt",
        "int8": directory / "best_int8_checkpoint.pt",
        "val_pred": directory / "validation_predictions.npz",
        "test_pred": directory / "test_predictions.npz",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    status = read_json(paths["status"])
    manifest = read_json(paths["manifest"])
    metrics = read_json(paths["metrics"])
    history = read_csv(paths["history"])
    qat_checkpoint = torch.load(paths["qat"], map_location="cpu", weights_only=False)
    int8_checkpoint = torch.load(paths["int8"], map_location="cpu", weights_only=False)

    source_model, symbol, source_path, source_hash, source_variant, pruning_metadata = build_source(
        architecture,
        variant,
        seed,
        model_module,
        pruning_module,
        b0_lookup,
        pruning_lookup,
    )

    prepared = qat_module.prepare_qat_model(source_model, backend)
    prepared.load_state_dict(qat_checkpoint["prepared_model_state_dict"], strict=True)
    prepared.eval()

    optimizer = torch.optim.AdamW(
        prepared.parameters(),
        lr=QAT_LR[architecture],
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.load_state_dict(qat_checkpoint["optimizer_state_dict"])

    int8_model = qat_module.convert_qat_model(prepared)
    int8_model.load_state_dict(int8_checkpoint["INT8_model_state_dict"], strict=True)
    int8_model.eval()

    with np.load(paths["val_pred"]) as z:
        val_true_saved = np.asarray(z["y_true"], dtype=np.int64)
        val_pred_saved = np.asarray(z["y_pred"])
        val_prob_saved = np.asarray(z["probabilities"])
        val_raw_saved = np.asarray(z["raw_row_count"], dtype=np.int64)

    with np.load(paths["test_pred"]) as z:
        test_true_saved = np.asarray(z["y_true"], dtype=np.int64)
        test_pred_saved = np.asarray(z["y_pred"])
        test_prob_saved = np.asarray(z["probabilities"])
        test_raw_saved = np.asarray(z["raw_row_count"], dtype=np.int64)

    val_pred_repeat, val_prob_repeat, repeat_seconds = infer_validation(
        int8_model,
        x_val,
        mean64,
        scale64,
        qat_module,
    )

    val_primary_repeat = metric_view(y_val, val_pred_repeat, val_prob_repeat, None)
    val_primary_saved = metric_view(val_true_saved, val_pred_saved, val_prob_saved, None)
    val_weighted_saved = metric_view(val_true_saved, val_pred_saved, val_prob_saved, val_raw_saved)
    test_primary_saved = metric_view(test_true_saved, test_pred_saved, test_prob_saved, None)
    test_weighted_saved = metric_view(test_true_saved, test_pred_saved, test_prob_saved, test_raw_saved)

    optimizer_group = optimizer.param_groups[0]

    checks = {
        "status_complete": status.get("status") == "completed" and status.get("stage") == "completed",
        "identity": (
            status.get("architecture") == architecture
            and status.get("variant") == variant
            and int(status.get("seed")) == seed
            and metrics.get("architecture") == architecture
            and metrics.get("variant") == variant
            and int(metrics.get("seed")) == seed
            and qat_checkpoint.get("architecture") == architecture
            and qat_checkpoint.get("variant") == variant
            and int(qat_checkpoint.get("seed")) == seed
            and int8_checkpoint.get("architecture") == architecture
            and int8_checkpoint.get("variant") == variant
            and int(int8_checkpoint.get("seed")) == seed
        ),
        "source_identity": (
            source_variant == SOURCE_VARIANT[variant]
            and source_variant == metrics["source"]["source_variant"]
            and source_variant == aggregate_row["source_variant"]
        ),
        "manifest_integrity": manifest.get("status") == "completed" and manifest.get("all_integrity_checks_passed") is True and manifest_artifacts_ok(directory, manifest),
        "hashes": (
            aggregate_row["metrics_sha256"] == sha(paths["metrics"])
            and aggregate_row["run_manifest_sha256"] == sha(paths["manifest"])
            and aggregate_row["QAT_checkpoint_sha256"] == sha(paths["qat"])
            and aggregate_row["INT8_checkpoint_sha256"] == sha(paths["int8"])
            and status.get("metrics_sha256") == sha(paths["metrics"])
            and status.get("run_manifest_sha256") == sha(paths["manifest"])
            and status.get("QAT_checkpoint_sha256") == sha(paths["qat"])
            and status.get("INT8_checkpoint_sha256") == sha(paths["int8"])
            and status.get("test_predictions_sha256") == sha(paths["test_pred"])
        ),
        "source_hash": (
            source_hash == preflight_row["source_checkpoint_sha256"]
            and source_hash == metrics["source"]["source_checkpoint_sha256"]
            and source_hash == aggregate_row["source_checkpoint_sha256"]
        ),
        "backend": (
            metrics["quantization"]["backend"] == backend
            and qat_checkpoint["QAT_backend"] == backend
            and int8_checkpoint["QAT_backend"] == backend
        ),
        "engine_version": (
            metrics["QAT_engine_version"] == QAT_ENGINE_VERSION
            and qat_checkpoint["QAT_engine_version"] == QAT_ENGINE_VERSION
            and int8_checkpoint["QAT_engine_version"] == QAT_ENGINE_VERSION
        ),
        "topology": (
            qat_module.float_linear_widths(source_model) == WIDTHS[architecture][variant]
            and qat_checkpoint["float_linear_widths"] == WIDTHS[architecture][variant]
            and int8_checkpoint["float_linear_widths"] == WIDTHS[architecture][variant]
        ),
        "all_linears_int8": (
            len(qat_module.static_quantized_linear_modules(int8_model)) == len(WIDTHS[architecture][variant])
            and all(dtype == "torch.qint8" for dtype in qat_module.quantized_weight_dtypes(int8_model))
        ),
        "training_parameters": (
            metrics["training"]["optimizer"] == "AdamW"
            and int(metrics["training"]["batch_size"]) == BATCH
            and int(metrics["training"]["max_epochs"]) == MAX_EPOCHS
            and int(metrics["training"]["early_stopping_patience"]) == PATIENCE
            and close(metrics["training"]["early_stopping_min_delta"], MIN_DELTA)
            and close(metrics["training"]["learning_rate"], QAT_LR[architecture])
            and close(metrics["training"]["weight_decay"], WEIGHT_DECAY)
            and close(matrix_row["learning_rate"], QAT_LR[architecture])
        ),
        "optimizer_state": (
            len(optimizer.state) > 0
            and close(optimizer_group["lr"], QAT_LR[architecture])
            and close(optimizer_group["weight_decay"], WEIGHT_DECAY)
        ),
        "history": (
            len(history) == int(metrics["training"]["epochs_completed"]) == int(aggregate_row["epochs_completed"])
            and 1 <= len(history) <= MAX_EPOCHS
            and history_ok(
                history,
                int(qat_checkpoint["best_epoch"]),
                float(qat_checkpoint["best_validation_INT8_macro_f1"]),
            )
        ),
        "selection": (
            int(qat_checkpoint["best_epoch"]) == int(int8_checkpoint["best_epoch"]) == int(metrics["selection"]["best_epoch"]) == int(aggregate_row["best_epoch"])
            and close(qat_checkpoint["best_validation_INT8_macro_f1"], int8_checkpoint["best_validation_INT8_macro_f1"])
            and close(qat_checkpoint["best_validation_INT8_macro_f1"], metrics["selection"]["best_validation_INT8_macro_f1"])
            and close(qat_checkpoint["best_validation_INT8_macro_f1"], aggregate_row["best_validation_INT8_macro_f1"])
        ),
        "saved_shapes": (
            val_true_saved.shape == (VAL_ROWS,)
            and val_pred_saved.shape == (VAL_ROWS,)
            and val_prob_saved.shape == (VAL_ROWS, CLASSES)
            and val_raw_saved.shape == (VAL_ROWS,)
            and test_true_saved.shape == (TEST_ROWS,)
            and test_pred_saved.shape == (TEST_ROWS,)
            and test_prob_saved.shape == (TEST_ROWS, CLASSES)
            and test_raw_saved.shape == (TEST_ROWS,)
        ),
        "saved_cache_identity": (
            np.array_equal(val_true_saved, y_val)
            and np.array_equal(val_raw_saved, raw_val)
            and np.array_equal(test_true_saved, y_test)
            and np.array_equal(test_raw_saved, raw_test)
        ),
        "probabilities": (
            np.isfinite(val_prob_saved).all()
            and np.isfinite(test_prob_saved).all()
            and np.allclose(val_prob_saved.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
            and np.allclose(test_prob_saved.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        ),
        "validation_exact": np.array_equal(val_pred_repeat, val_pred_saved) and np.array_equal(val_prob_repeat, val_prob_saved),
        "metrics_exact": (
            metrics_equal(val_primary_repeat, metrics["validation"]["primary_fingerprint_level"])
            and metrics_equal(val_primary_saved, metrics["validation"]["primary_fingerprint_level"])
            and metrics_equal(val_weighted_saved, metrics["validation"]["secondary_raw_record_weighted"])
            and metrics_equal(test_primary_saved, metrics["test"]["primary_fingerprint_level"])
            and metrics_equal(test_weighted_saved, metrics["test"]["secondary_raw_record_weighted"])
        ),
        "aggregate_metrics": (
            close(aggregate_row["test_fingerprint_macro_f1"], test_primary_saved["macro_f1"])
            and close(aggregate_row["test_fingerprint_accuracy"], test_primary_saved["accuracy"])
            and close(aggregate_row["test_raw_weighted_macro_f1"], test_weighted_saved["macro_f1"])
            and close(aggregate_row["test_gafgyt_fnr"], test_primary_saved["per_class"]["gafgyt"]["false_negative_rate"])
            and close(aggregate_row["test_mirai_fnr"], test_primary_saved["per_class"]["mirai"]["false_negative_rate"])
        ),
        "test_policy": (
            int(status["test_evaluation_count"]) == 1
            and int(metrics["test_evaluation_count"]) == 1
            and int(metrics["data_access"]["test_evaluation_count"]) == 1
            and int(manifest["fit_scope"]["test_evaluation_count"]) == 1
            and int(aggregate_row["test_evaluation_count"]) == 1
            and metrics["selection"]["test_used_for_selection"] is False
            and metrics["data_access"]["test_used_for_selection"] is False
            and manifest["fit_scope"]["test_used_for_selection"] is False
        ),
        "fragility": (
            bool(metrics["fragility_gate"]["triggered"])
            == bool(status["fragility_gate_triggered"])
            == parse_bool(aggregate_row["fragility_gate_triggered"])
            == False
        ),
    }

    bad = [name for name, passed in checks.items() if not passed]
    if bad:
        raise RuntimeError(
            f"{architecture} {variant} seed {seed} verification failed: "
            + ", ".join(bad)
        )

    verified.append(
        {
            "architecture": architecture,
            "variant": variant,
            "source_variant": source_variant,
            "seed": seed,
            "best_epoch": int(qat_checkpoint["best_epoch"]),
            "epochs_completed": len(history),
            "best_validation_INT8_macro_f1": val_primary_repeat["macro_f1"],
            "test_fingerprint_macro_f1": test_primary_saved["macro_f1"],
            "test_fingerprint_accuracy": test_primary_saved["accuracy"],
            "test_raw_weighted_macro_f1": test_weighted_saved["macro_f1"],
            "test_gafgyt_fnr": test_primary_saved["per_class"]["gafgyt"]["false_negative_rate"],
            "test_mirai_fnr": test_primary_saved["per_class"]["mirai"]["false_negative_rate"],
            "INT8_to_float_state_size_ratio": float(metrics["model_complexity"]["INT8_to_float_state_size_ratio"]),
            "validation_repeat_seconds": repeat_seconds,
            "test_evaluation_count": 1,
            "test_model_inference_repeated": False,
            "fragility_gate_triggered": False,
            "run_directory": str(directory),
            "QAT_checkpoint_sha256": sha(paths["qat"]),
            "INT8_checkpoint_sha256": sha(paths["int8"]),
            "metrics_sha256": sha(paths["metrics"]),
            "run_manifest_sha256": sha(paths["manifest"]),
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/30] {architecture} seed={seed} {variant} | "
        "QAT checkpoint=loaded | INT8 checkpoint=loaded | "
        "validation=exact | test inference=False | fragility=False | all checks=True",
        flush=True,
    )

fields = list(verified[0].keys())
atomic_csv(OUT_RUNS, verified, fields)

group_lookup = {(r["architecture"], r["variant"]): r for r in groups_rows}
verified_groups: list[dict[str, Any]] = []

for architecture in ARCHS:
    for variant in VARIANTS:
        rows = [r for r in verified if r["architecture"] == architecture and r["variant"] == variant]
        saved = group_lookup[(architecture, variant)]
        f1_stats = stats([float(r["test_fingerprint_macro_f1"]) for r in rows])
        raw_stats = stats([float(r["test_raw_weighted_macro_f1"]) for r in rows])
        val_stats = stats([float(r["best_validation_INT8_macro_f1"]) for r in rows])
        ratio_stats = stats([float(r["INT8_to_float_state_size_ratio"]) for r in rows])

        checks = {
            "count": len(rows) == 5 and int(saved["run_count"]) == 5,
            "source": saved["source_variant"] == SOURCE_VARIANT[variant],
            "val_mean": close(saved["mean_best_validation_INT8_macro_f1"], val_stats["mean"]),
            "f1_mean": close(saved["mean_test_fingerprint_macro_f1"], f1_stats["mean"]),
            "f1_std": close(saved["std_test_fingerprint_macro_f1"], f1_stats["std_population"]),
            "f1_min": close(saved["min_test_fingerprint_macro_f1"], f1_stats["minimum"]),
            "f1_max": close(saved["max_test_fingerprint_macro_f1"], f1_stats["maximum"]),
            "raw_mean": close(saved["mean_test_raw_weighted_macro_f1"], raw_stats["mean"]),
            "ratio_mean": close(saved["mean_INT8_to_float_state_size_ratio"], ratio_stats["mean"]),
            "test_count": int(saved["test_evaluation_count_per_run"]) == 1,
            "fragility": parse_bool(saved["fragility_triggered_in_any_run"]) is False,
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise RuntimeError(f"{architecture} {variant} group verification failed: " + ", ".join(bad))

        verified_groups.append(
            {
                "architecture": architecture,
                "variant": variant,
                "source_variant": SOURCE_VARIANT[variant],
                "run_count": 5,
                "validation_macro_f1": val_stats,
                "test_macro_f1": f1_stats,
                "test_raw_weighted_macro_f1": raw_stats,
                "INT8_to_float_state_size_ratio": ratio_stats,
                "fragility_triggered_in_any_run": False,
                "all_checks_passed": True,
            }
        )

global_checks = {
    "thirty_runs": len(verified) == 30,
    "six_groups": len(verified_groups) == 6,
    "all_runs_passed": all(r["all_checks_passed"] for r in verified),
    "all_groups_passed": all(g["all_checks_passed"] for g in verified_groups),
    "test_count_one": all(r["test_evaluation_count"] == 1 for r in verified),
    "test_not_repeated": all(r["test_model_inference_repeated"] is False for r in verified),
    "no_fragility": all(r["fragility_gate_triggered"] is False for r in verified),
}
failed = [k for k, v in global_checks.items() if not v]
if failed:
    raise RuntimeError("QAT global verification failed: " + ", ".join(failed))

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": "QAT_P25_QAT_P50_QAT_independent_verification",
    "protocol_version": PROTOCOL_VERSION,
    "QAT_engine_version": QAT_ENGINE_VERSION,
    "verified_at_utc": now(),
    "run_count": 30,
    "group_count": 6,
    "QAT_backend": backend,
    "verification_policy": {
        "QAT_checkpoint_loading_performed": True,
        "INT8_checkpoint_loading_performed": True,
        "optimizer_state_loading_performed": True,
        "independent_validation_inference_repeated": True,
        "test_model_inference_repeated": False,
        "saved_test_metrics_recomputed": True,
        "test_evaluation_count_verified_per_run": 1,
    },
    "verified_runs_csv": record(OUT_RUNS),
    "verified_groups": verified_groups,
    "entry_checks": entry_checks,
    "cache_checks": cache_checks,
    "global_checks": global_checks,
    "ready_for_PTQ_branch": True,
    "all_checks_passed": True,
}
atomic_json(OUT_VERIFY, verification)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": "QAT_P25_QAT_P50_QAT_evaluation",
    "protocol_version": PROTOCOL_VERSION,
    "QAT_engine_version": QAT_ENGINE_VERSION,
    "locked_at_utc": now(),
    "run_count": 30,
    "group_count": 6,
    "QAT_backend": backend,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "fragility_triggered_in_any_run": False,
    "verified_runs_csv": str(OUT_RUNS),
    "verified_runs_csv_sha256": sha(OUT_RUNS),
    "verification_report": str(OUT_VERIFY),
    "verification_report_sha256": sha(OUT_VERIFY),
    "aggregate_runs_csv": str(RUNS_CSV),
    "aggregate_runs_csv_sha256": sha(RUNS_CSV),
    "group_summary_csv": str(GROUPS_CSV),
    "group_summary_csv_sha256": sha(GROUPS_CSV),
    "aggregate_json": str(SUMMARY_JSON),
    "aggregate_json_sha256": sha(SUMMARY_JSON),
    "ready_for_PTQ_branch": True,
    "all_checks_passed": True,
}
atomic_json(OUT_LOCK, lock)

manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": "QAT_P25_QAT_P50_QAT_evaluation",
    "protocol_version": PROTOCOL_VERSION,
    "locked_at_utc": now(),
    "source_artifacts": [
        record(PROTOCOL),
        record(MATRIX),
        record(B0_PAIR_LOCK),
        record(B0_REGISTRY),
        record(PRUNING_ENGINE_LOCK),
        record(PRUNING_ENGINE),
        record(PRUNING_SOURCES_LOCK),
        record(PRUNING_REGISTRY),
        record(QAT_PREFLIGHT_LOCK),
        record(QAT_PREFLIGHT_MATRIX),
        record(MODEL_SOURCE),
        record(QAT_ENGINE),
        record(SCALER),
        record(RUNS_CSV),
        record(GROUPS_CSV),
        record(SUMMARY_JSON),
        record(COMPLETION_JSON),
    ],
    "generated_artifacts": [
        record(OUT_RUNS),
        record(OUT_VERIFY),
        record(OUT_LOCK),
    ],
    "run_count": 30,
    "group_count": 6,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "ready_for_PTQ_branch": True,
    "all_checks_passed": True,
}
atomic_json(OUT_MANIFEST, manifest)

print()
print("=" * 92)
print("PHASE 5 QAT VERIFICATION SUMMARY")
print("=" * 92)
print("Runs independently verified     : 30")
print("Groups independently verified   : 6")
print("QAT checkpoint loading          : PASSED")
print("INT8 checkpoint loading         : PASSED")
print("Optimizer-state loading         : PASSED")
print("Independent validation inference: PASSED")
print("Saved test metrics recomputed   : PASSED")
print("Test model inference repeated   : False")
print("Test evaluation count per run   : 1")
print("Fragility after QAT             : False")
print("QAT evaluation status           : LOCKED")
print("Ready for PTQ branch            : True")
print(f"Verification report             : {OUT_VERIFY}")
print(f"Lock file                       : {OUT_LOCK}")
print("All checks passed               : True")
print("PHASE 5 QAT, P25-QAT AND P50-QAT VERIFIED AND LOCKED")
