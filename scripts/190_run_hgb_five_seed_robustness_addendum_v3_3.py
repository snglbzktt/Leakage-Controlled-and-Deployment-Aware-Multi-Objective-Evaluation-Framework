from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Match the original HGB execution environment as closely as possible.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path.cwd()

CACHE_ROOT = ROOT / "data" / "cache" / "tabular_baseline_family3_v3_2"
ORIGINAL_RUN = (
    ROOT / "results" / "v2" / "tabular_baselines" / "runs"
    / "hist_gradient_boosting_b0__seed_2026"
)
OUT_ROOT = ROOT / "results" / "v2" / "hgb_five_seed_robustness_v3_3"
AUDIT_ROOT = ROOT / "results" / "v2" / "audit"

SUMMARY_JSON = AUDIT_ROOT / "hgb_five_seed_robustness_v3_3_summary.json"
RUNS_CSV = AUDIT_ROOT / "hgb_five_seed_robustness_v3_3_runs.csv"

SEEDS = [42, 123, 2026, 3407, 8192]
EXPECTED_SKLEARN = "1.9.0"
CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
CHUNK_SIZE = 100_000

LOCKED_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0001,
    "max_bins": 255,
    "class_weight": "balanced",
    "early_stopping": False,
}

def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        sample_weight=sample_weight,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        sample_weight=sample_weight,
    )
    if sample_weight is not None:
        cm = np.rint(cm).astype(np.int64)
    else:
        cm = cm.astype(np.int64)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }
    for i, name in enumerate(CLASS_NAMES):
        result["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
            "fnr": float(1.0 - recall[i]),
        }
    return result

def predict_chunked(model, x: np.ndarray) -> np.ndarray:
    pred = np.empty(len(x), dtype=np.int8)
    for start in range(0, len(x), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(x))
        xb = np.asarray(x[start:end], dtype=np.float32)
        if not np.isfinite(xb).all():
            raise RuntimeError(f"Non-finite values in X_test[{start}:{end}]")
        pred[start:end] = model.predict(xb).astype(np.int8)
    return pred

def run_seed(seed: int, x_train, y_train, x_test, y_test, raw_counts) -> dict[str, Any]:
    seed_dir = OUT_ROOT / f"seed_{seed}"
    result_path = seed_dir / "metrics.json"

    # Resume-safe: if a completed result exists, reuse it.
    if result_path.exists():
        obj = json.loads(result_path.read_text(encoding="utf-8"))
        if obj.get("status") == "completed" and obj.get("seed") == seed:
            print(f"[resume] seed={seed} -> existing completed result")
            return obj

    seed_dir.mkdir(parents=True, exist_ok=True)

    model = HistGradientBoostingClassifier(
        loss=LOCKED_PARAMETERS["loss"],
        learning_rate=LOCKED_PARAMETERS["learning_rate"],
        max_iter=LOCKED_PARAMETERS["max_iter"],
        max_leaf_nodes=LOCKED_PARAMETERS["max_leaf_nodes"],
        max_depth=LOCKED_PARAMETERS["max_depth"],
        min_samples_leaf=LOCKED_PARAMETERS["min_samples_leaf"],
        l2_regularization=LOCKED_PARAMETERS["l2_regularization"],
        max_bins=LOCKED_PARAMETERS["max_bins"],
        class_weight=LOCKED_PARAMETERS["class_weight"],
        early_stopping=LOCKED_PARAMETERS["early_stopping"],
        random_state=seed,
        verbose=0,
    )

    print("=" * 88)
    print(f"HGB reviewer-driven five-seed robustness addendum | seed={seed}")
    print("=" * 88)

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = predict_chunked(model, x_test)
    test_seconds = time.perf_counter() - t1

    primary = compute_metrics(y_test, y_pred, sample_weight=None)
    weighted = compute_metrics(y_test, y_pred, sample_weight=raw_counts)

    obj = {
        "status": "completed",
        "purpose": "reviewer_driven_hgb_seed_stability_addendum",
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "input_space": "canonical_float32_unscaled",
        "test_evaluation_count": 1,
        "parameters": {**LOCKED_PARAMETERS, "random_state": seed},
        "fit_seconds": float(fit_seconds),
        "test_seconds": float(test_seconds),
        "n_iter": int(model.n_iter_),
        "do_early_stopping": bool(model.do_early_stopping_),
        "test_fingerprint": primary,
        "test_occurrence_weighted": weighted,
    }
    atomic_json(result_path, obj)
    return obj

def main() -> None:
    required = [
        CACHE_ROOT / "X_train.npy",
        CACHE_ROOT / "y_train.npy",
        CACHE_ROOT / "X_test.npy",
        CACHE_ROOT / "y_test.npy",
        CACHE_ROOT / "raw_row_count_test.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required cache files:\n" + "\n".join(missing))

    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"Expected scikit-learn {EXPECTED_SKLEARN}, found {sklearn.__version__}. "
            "Run this in the same project venv used for the paper."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    x_train = np.load(CACHE_ROOT / "X_train.npy", mmap_mode="r")
    y_train = np.asarray(np.load(CACHE_ROOT / "y_train.npy", mmap_mode="r"), dtype=np.int64)
    x_test = np.load(CACHE_ROOT / "X_test.npy", mmap_mode="r")
    y_test = np.asarray(np.load(CACHE_ROOT / "y_test.npy", mmap_mode="r"), dtype=np.int64)
    raw_counts = np.asarray(
        np.load(CACHE_ROOT / "raw_row_count_test.npy", mmap_mode="r"),
        dtype=np.int64,
    )

    print(f"train fingerprints: {len(y_train):,}")
    print(f"test fingerprints : {len(y_test):,}")
    print(f"seeds             : {SEEDS}")
    print(f"sklearn           : {sklearn.__version__}")
    print("early_stopping    : False")
    print()

    runs = [run_seed(s, x_train, y_train, x_test, y_test, raw_counts) for s in SEEDS]

    macro = np.asarray([r["test_fingerprint"]["macro_f1"] for r in runs], dtype=float)
    acc = np.asarray([r["test_fingerprint"]["accuracy"] for r in runs], dtype=float)
    occ = np.asarray([r["test_occurrence_weighted"]["macro_f1"] for r in runs], dtype=float)
    gaf = np.asarray([r["test_fingerprint"]["per_class"]["gafgyt"]["fnr"] for r in runs], dtype=float)
    mir = np.asarray([r["test_fingerprint"]["per_class"]["mirai"]["fnr"] for r in runs], dtype=float)

    def stats(a: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(a)),
            "std_population": float(np.std(a, ddof=0)),
            "std_sample": float(np.std(a, ddof=1)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    summary = {
        "status": "completed",
        "purpose": "reviewer_driven_hgb_seed_stability_addendum",
        "interpretation": (
            "Post-hoc robustness extension only. Hyperparameters, data splits, "
            "input space, and test metric definitions are unchanged. No seed is "
            "selected by performance; all five predefined seeds are reported."
        ),
        "seeds": SEEDS,
        "run_count": len(runs),
        "sklearn_version": sklearn.__version__,
        "parameters": LOCKED_PARAMETERS,
        "metrics": {
            "test_fingerprint_macro_f1": stats(macro),
            "test_fingerprint_accuracy": stats(acc),
            "test_occurrence_weighted_macro_f1": stats(occ),
            "test_gafgyt_fnr": stats(gaf),
            "test_mirai_fnr": stats(mir),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "fingerprint_macro_f1": r["test_fingerprint"]["macro_f1"],
                "fingerprint_accuracy": r["test_fingerprint"]["accuracy"],
                "occurrence_weighted_macro_f1": r["test_occurrence_weighted"]["macro_f1"],
                "gafgyt_fnr": r["test_fingerprint"]["per_class"]["gafgyt"]["fnr"],
                "mirai_fnr": r["test_fingerprint"]["per_class"]["mirai"]["fnr"],
                "fit_seconds": r["fit_seconds"],
                "n_iter": r["n_iter"],
            }
            for r in runs
        ],
    }

    # Cross-check the new seed-2026 result against the original paper value.
    paper_hgb_macro = 0.9997380683535377
    seed2026 = next(x for x in summary["per_seed"] if x["seed"] == 2026)
    summary["seed2026_crosscheck"] = {
        "paper_value": paper_hgb_macro,
        "new_value": seed2026["fingerprint_macro_f1"],
        "abs_difference": abs(seed2026["fingerprint_macro_f1"] - paper_hgb_macro),
        "matches_within_1e_10": abs(seed2026["fingerprint_macro_f1"] - paper_hgb_macro) <= 1e-10,
    }

    atomic_json(SUMMARY_JSON, summary)

    import csv
    with RUNS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "fingerprint_macro_f1",
                "fingerprint_accuracy",
                "occurrence_weighted_macro_f1",
                "gafgyt_fnr",
                "mirai_fnr",
                "fit_seconds",
                "n_iter",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["per_seed"])

    print()
    print("=" * 88)
    print("HGB FIVE-SEED ROBUSTNESS SUMMARY")
    print("=" * 88)
    print(f"Macro-F1 mean       : {summary['metrics']['test_fingerprint_macro_f1']['mean']:.9f}")
    print(f"Macro-F1 std(pop)   : {summary['metrics']['test_fingerprint_macro_f1']['std_population']:.9f}")
    print(f"Macro-F1 min        : {summary['metrics']['test_fingerprint_macro_f1']['min']:.9f}")
    print(f"Macro-F1 max        : {summary['metrics']['test_fingerprint_macro_f1']['max']:.9f}")
    print(f"Seed-2026 crosscheck: {summary['seed2026_crosscheck']['matches_within_1e_10']}")
    print(f"Summary JSON        : {SUMMARY_JSON}")
    print(f"Runs CSV            : {RUNS_CSV}")

if __name__ == "__main__":
    main()
