
"""Run the N-BaIoT duplicate-leakage ablation with the locked B0 model.

This script compares the row-wise (naive_row) and duplicate-group-controlled
(grouped) arms produced by stage 35.  Each arm gets its own train-only scaler
and class weights.  The B0 ``tinyml_mlp`` is then trained with the five locked
training seeds and evaluated once on the corresponding test split.

The materialized cache is intentionally resumable because it is several GB.
Delete only ``data/processed/vs1_leakage_ablation`` to rebuild that cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nbaiot_models import create_model  # noqa: E402


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "splits" / "vs1_leakage_ablation_seed2026.sqlite"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "protocols" / "nbaiot_family3_fp32_baseline_protocol_v1.json"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "processed" / "vs1_leakage_ablation"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "vs1" / "leakage_ablation"
ARMS = ("naive_row", "grouped")
SPLITS = ("train", "validation", "test")
SEEDS = (42, 123, 2026, 3407, 8192)
CLASS_NAMES = ("benign", "gafgyt", "mirai")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 36: leakage ablation training")
    p.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--overwrite-results", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def family_label(raw: str) -> int:
    value = raw.lower()
    if "benign" in value:
        return 0
    if "gafgyt" in value:
        return 1
    if "mirai" in value:
        return 2
    raise ValueError(f"Bilinmeyen family_3 sınıf etiketi: {raw!r}")


def source_database_path(database: Path) -> Path:
    with sqlite3.connect(database) as con:
        row = con.execute("SELECT value FROM metadata WHERE key='source_database'").fetchone()
    if row is None:
        raise RuntimeError("Stage-35 veritabanında source_database metadata kaydı yok.")
    path = Path(str(row[0]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Parmak izi kaynak veritabanı bulunamadı: {path}")
    return path.resolve()


def discover_files(source_db: Path) -> list[tuple[int, Path, str]]:
    """Discover file id/path/label without assuming the path column's name."""
    with sqlite3.connect(source_db) as con:
        columns = [str(r[1]) for r in con.execute("PRAGMA table_info(files)")]
        path_candidates = ("file_path", "csv_path", "path", "relative_path", "source_file")
        path_col = next((x for x in path_candidates if x in columns), None)
        if path_col is None or "file_id" not in columns or "class_label" not in columns:
            raise RuntimeError(f"source.files şeması desteklenmiyor. Sütunlar: {columns}")
        rows = con.execute(
            f'SELECT file_id, "{path_col}", class_label FROM files ORDER BY file_id'
        ).fetchall()
    result: list[tuple[int, Path, str]] = []
    for file_id, raw_path, label in rows:
        path = Path(str(raw_path))
        candidates = [path]
        if not path.is_absolute():
            candidates += [
    PROJECT_ROOT / path,
    PROJECT_ROOT / "data" / "interim" / "nbaiot_unpacked" / path,
    source_db.parent / path,
]
        resolved = next((p.resolve() for p in candidates if p.exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"CSV bulunamadı (file_id={file_id}): {raw_path}")
        result.append((int(file_id), resolved, str(label)))
    return result


def counts(database: Path, arm: str) -> dict[str, int]:
    column = "naive_row_split" if arm == "naive_row" else "grouped_split"
    with sqlite3.connect(database) as con:
        rows = con.execute(
            f"SELECT {column}, COUNT(*) FROM assignments GROUP BY {column}"
        ).fetchall()
    out = {str(k): int(v) for k, v in rows}
    if set(out) != set(SPLITS):
        raise RuntimeError(f"{arm} split kümesi geçersiz: {out}")
    return out


def selected_rows(database: Path, file_id: int, arm: str, split: str) -> np.ndarray:
    column = "naive_row_split" if arm == "naive_row" else "grouped_split"
    with sqlite3.connect(database) as con:
        rows = con.execute(
            f"SELECT row_number FROM assignments WHERE {column}=? AND file_id=? ORDER BY row_number",
            (split, file_id),
        ).fetchall()
    return np.fromiter(
    (int(r[0]) - 1 for r in rows),
    dtype=np.int64,
    count=len(rows),
)


def materialize_arm(database: Path, arm: str, cache: Path, chunk_size: int) -> dict[str, Any]:
    arm_dir = cache / arm
    manifest_file = arm_dir / "manifest.json"
    if manifest_file.exists():
        with manifest_file.open(encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("status") == "completed":
            print(f"[{arm}] Hazır önbellek kullanılıyor.", flush=True)
            return manifest

    arm_dir.mkdir(parents=True, exist_ok=True)
    source_db = source_database_path(database)
    files = discover_files(source_db)
    split_counts = counts(database, arm)
    feature_count: int | None = None
    arrays: dict[str, tuple[np.memmap, np.memmap]] = {}
    offsets = {s: 0 for s in SPLITS}

    # Read the first CSV only to establish dimensionality.
    first = pd.read_csv(files[0][1], nrows=2)
    feature_count = int(first.shape[1])
    if feature_count != 115:
        raise RuntimeError(f"Beklenen 115 özellik, bulunan {feature_count}.")
    for split in SPLITS:
        x = np.lib.format.open_memmap(
            arm_dir / f"X_{split}.npy", mode="w+", dtype=np.float32,
            shape=(split_counts[split], feature_count),
        )
        y = np.lib.format.open_memmap(
            arm_dir / f"y_{split}.npy", mode="w+", dtype=np.int64,
            shape=(split_counts[split],),
        )
        arrays[split] = (x, y)

    for file_no, (file_id, csv_path, raw_label) in enumerate(files, start=1):
        row_sets = {s: selected_rows(database, file_id, arm, s) for s in SPLITS}
        pointers = {s: 0 for s in SPLITS}
        base = 0
        target = family_label(raw_label)
        for frame in pd.read_csv(csv_path, chunksize=chunk_size):
            values = frame.to_numpy(dtype=np.float32, copy=False)
            if values.shape[1] != feature_count or not np.isfinite(values).all():
                raise RuntimeError(f"Geçersiz CSV özellikleri: {csv_path}")
            end = base + len(values)
            for split in SPLITS:
                chosen = row_sets[split]
                left = int(np.searchsorted(chosen, base, side="left"))
                right = int(np.searchsorted(chosen, end, side="left"))
                local = chosen[left:right] - base
                n = len(local)
                if n:
                    out = offsets[split]
                    arrays[split][0][out:out+n] = values[local]
                    arrays[split][1][out:out+n] = target
                    offsets[split] += n
                pointers[split] = right
            base = end
        print(f"[{arm}] Dosya {file_no}/{len(files)} hazırlandı: {csv_path.name}", flush=True)

    for split in SPLITS:
        if offsets[split] != split_counts[split]:
            raise RuntimeError(f"{arm}/{split} satır uyuşmazlığı: {offsets[split]} != {split_counts[split]}")
        arrays[split][0].flush(); arrays[split][1].flush()

    x_train = np.load(arm_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(arm_dir / "y_train.npy", mmap_mode="r")
    total = np.zeros(feature_count, dtype=np.float64)
    square = np.zeros(feature_count, dtype=np.float64)
    for start in range(0, len(x_train), chunk_size):
        part = np.asarray(x_train[start:start+chunk_size], dtype=np.float64)
        total += part.sum(axis=0); square += np.square(part).sum(axis=0)
    mean = total / len(x_train)
    var = np.maximum(square / len(x_train) - np.square(mean), 0.0)
    std = np.sqrt(var); std[std < 1e-12] = 1.0
    class_counts = np.bincount(y_train, minlength=3)
    weights = len(y_train) / (3.0 * class_counts)
    np.save(arm_dir / "mean.npy", mean.astype(np.float32))
    np.save(arm_dir / "std.npy", std.astype(np.float32))
    np.save(arm_dir / "class_weights.npy", weights.astype(np.float32))
    manifest = {
        "status": "completed", "created_at_utc": utc_now(), "arm": arm,
        "feature_count": feature_count, "class_names": list(CLASS_NAMES),
        "split_counts": split_counts, "train_class_counts": class_counts.tolist(),
        "scaler_fit_split": "train", "class_weights_fit_split": "train",
    }
    atomic_json(manifest_file, manifest)
    return manifest


class MemmapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, directory: Path, split: str) -> None:
        self.x = np.load(directory / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(directory / f"y_{split}.npy", mmap_mode="r")
        self.mean = np.load(directory / "mean.npy")
        self.std = np.load(directory / "std.npy")
    def __len__(self) -> int:
        return len(self.y)
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (np.asarray(self.x[i], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(x.copy()), torch.tensor(int(self.y[i]), dtype=torch.long)


def reproducible(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def confusion_metrics(cm: np.ndarray) -> dict[str, Any]:
    support = cm.sum(axis=1); predicted = cm.sum(axis=0); tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall) != 0)
    return {
        "accuracy": float(tp.sum()/cm.sum()), "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
        "per_class": {CLASS_NAMES[i]: {"support": int(support[i]), "precision": float(precision[i]),
                      "recall": float(recall[i]), "f1": float(f1[i]),
                      "false_negative_rate": float(1-recall[i])} for i in range(3)},
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, Any]:
    model.eval(); cm = np.zeros((3, 3), dtype=np.int64); loss_sum = 0.0; count = 0
    for x, y in loader:
        logits = model(x); loss = criterion(logits, y)
        pred = logits.argmax(1); n = len(y); loss_sum += float(loss)*n; count += n
        np.add.at(cm, (y.numpy(), pred.numpy()), 1)
    return {"loss": loss_sum/count, "sample_count": count, **confusion_metrics(cm)}


def protocol_settings(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        p = json.load(f)
    common = p["common_training_configuration"]
    model = p["model_specific_configuration"]["tinyml_mlp"]
    return {
        "batch_size": int(common["batch_size"]), "num_workers": int(common["num_workers"]),
        "max_epochs": int(common["max_epochs"]), "learning_rate": float(model["learning_rate"]),
        "weight_decay": float(common["weight_decay"]),
        "gradient_clip_norm": float(common["gradient_clip_norm"]),
        "patience": int(common["early_stopping_patience"]),
        "min_delta": float(common["early_stopping_min_delta"]),
        "torch_threads": int(common["torch_threads"]),
    }


def train_one(arm_dir: Path, output: Path, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reproducible(seed, cfg["torch_threads"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(MemmapDataset(arm_dir, "train"), batch_size=cfg["batch_size"],
                              shuffle=True, generator=generator, num_workers=cfg["num_workers"])
    val_loader = DataLoader(MemmapDataset(arm_dir, "validation"), batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(MemmapDataset(arm_dir, "test"), batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg["num_workers"])
    model = create_model("tinyml_mlp", input_features=115, num_classes=3)
    weights = torch.from_numpy(np.load(arm_dir / "class_weights.npy"))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best = -math.inf; best_epoch = 0; stale = 0; history = []
    checkpoint = output / "best_checkpoint.pt"
    for epoch in range(1, cfg["max_epochs"]+1):
        model.train(); running = 0.0; seen = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(x); loss = criterion(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step(); running += float(loss.detach())*len(y); seen += len(y)
        val = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": running/seen, **{f"validation_{k}": v for k,v in val.items() if k not in ("confusion_matrix","per_class")}})
        print(f"  seed={seed} epoch={epoch}: val_macro_f1={val['macro_f1']:.6f}", flush=True)
        if val["macro_f1"] > best + cfg["min_delta"]:
            best = val["macro_f1"]; best_epoch = epoch; stale = 0
            torch.save({"model_state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= cfg["patience"]: break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    test = evaluate(model, test_loader, criterion)
    result = {"completed_at_utc": utc_now(), "seed": seed, "best_epoch": best_epoch,
              "best_validation_macro_f1": best, "test": test, "history": history,
              "test_evaluation_count": 1, "test_used_for_selection": False}
    atomic_json(output / "result.json", result)
    return result


def main() -> None:
    args = parse_args(); start = time.monotonic()
    if not args.database.exists(): raise FileNotFoundError(args.database)
    if not args.protocol.exists(): raise FileNotFoundError(args.protocol)
    if tuple(args.seeds) != SEEDS:
        print("UYARI: Kilitli beş-seed kümesinden farklı bir küme istendi.", flush=True)
    manifests = {}
    for arm in args.arms:
        manifests[arm] = materialize_arm(args.database.resolve(), arm, args.cache_directory.resolve(), args.chunk_size)
    if args.prepare_only:
        print("Önbellek hazırlama tamamlandı; eğitim --prepare-only nedeniyle atlandı."); return
    cfg = protocol_settings(args.protocol)
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            run_dir = args.output_directory / arm / f"seed{seed}"
            result_file = run_dir / "result.json"
            if result_file.exists() and not args.overwrite_results:
                with result_file.open(encoding="utf-8") as f: result = json.load(f)
                print(f"[{arm}/seed{seed}] Tamamlanmış sonuç kullanılıyor.", flush=True)
            else:
                print(f"[{arm}/seed{seed}] Eğitim başlıyor...", flush=True)
                result = train_one(args.cache_directory / arm, run_dir, seed, cfg)
            rows.append({"arm": arm, "seed": seed, "best_epoch": result["best_epoch"],
                         **{f"test_{k}": v for k,v in result["test"].items() if isinstance(v,(int,float))}})
            gc.collect()
    frame = pd.DataFrame(rows); args.output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_directory / "leakage_ablation_runs.csv", index=False)
    metrics = ["test_accuracy", "test_balanced_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1"]
    aggregate = frame.groupby("arm")[metrics].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(c).rstrip("_") if isinstance(c,tuple) else c for c in aggregate.columns]
    aggregate.to_csv(args.output_directory / "leakage_ablation_aggregate.csv", index=False)
    atomic_json(args.output_directory / "summary.json", {"completed_at_utc": utc_now(),
                "model": "tinyml_mlp", "training_seeds": args.seeds, "arms": args.arms,
                "elapsed_minutes": (time.monotonic()-start)/60, "data_manifests": manifests,
                "test_evaluation_per_arm_seed": 1})
    print("="*78); print("36. aşama tamamlandı.")
    print(aggregate.to_string(index=False)); print("="*78)


if __name__ == "__main__":
    main()