from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

SMOKE_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "smoke"
    / "early_lodo"
    / "danmini_tinyml_mlp_smoke_v2.json"
)

SMOKE_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "smoke"
    / "early_lodo"
    / "danmini_tinyml_mlp_smoke_v2.txt"
)

SMOKE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "69_run_early_lodo_smoke_test_v2.py"
)

FEATURE_ALIGNMENT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_feature_alignment_summary_v2.json"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

LOCKED_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_smoke_test_locked_summary_v2.json"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_smoke_test_release_manifest_v2.csv"
)

COMPLETION_NOTE = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "PHASE_2F_EARLY_LODO_SMOKE_TEST_COMPLETE.md"
)


EXPECTED_SAMPLE_SIZES = {
    "train": 12_288,
    "validation": 6_144,
    "test": 6_144,
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def save_json(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


required_files = [
    SMOKE_JSON,
    SMOKE_REPORT,
    SMOKE_SCRIPT,
    FEATURE_ALIGNMENT_SUMMARY,
    PROTOCOL_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


smoke = json.loads(
    SMOKE_JSON.read_text(
        encoding="utf-8-sig",
    )
)

feature_alignment = json.loads(
    FEATURE_ALIGNMENT_SUMMARY.read_text(
        encoding="utf-8-sig",
    )
)

protocol = json.loads(
    PROTOCOL_FILE.read_text(
        encoding="utf-8-sig",
    )
)


smoke_checks = smoke.get(
    "validation_checks",
    {},
)

sample_sizes = smoke.get(
    "sample_sizes",
    {},
)

overlap_counts = smoke.get(
    "overlap_counts",
    {},
)

losses = smoke.get(
    "losses",
    {},
)

validation_metrics = smoke.get(
    "validation_metrics",
    {},
)

test_metrics = smoke.get(
    "test_metrics",
    {},
)


numeric_values = [
    losses.get("training"),
    losses.get("validation"),
    losses.get("test"),
    validation_metrics.get(
        "macro_f1_present_classes"
    ),
    test_metrics.get(
        "macro_f1_present_classes"
    ),
]


validation_checks = {
    "smoke_status_passed":
        smoke.get("status")
        == "passed",

    "smoke_all_checks_passed":
        smoke.get(
            "all_checks_passed"
        )
        is True,

    "all_smoke_internal_checks_true":
        bool(smoke_checks)
        and all(
            value is True
            for value
            in smoke_checks.values()
        ),

    "not_scientific_result":
        smoke.get(
            "scientific_result"
        )
        is False,

    "held_out_device_danmini":
        smoke.get(
            "held_out_device"
        )
        == "Danmini_Doorbell",

    "model_tinyml_mlp":
        smoke.get(
            "model_name"
        )
        == "tinyml_mlp",

    "seed_2026":
        int(
            smoke.get(
                "seed",
                -1,
            )
        )
        == 2026,

    "one_epoch":
        int(
            smoke.get(
                "epoch_count",
                -1,
            )
        )
        == 1,

    "sample_sizes_match":
        all(
            int(
                sample_sizes.get(
                    name,
                    -1,
                )
            )
            == expected
            for name, expected
            in EXPECTED_SAMPLE_SIZES.items()
        ),

    "all_overlap_counts_zero":
        overlap_counts
        == {
            "train_test": 0,
            "validation_test": 0,
            "train_validation": 0,
        },

    "model_parameter_count_9603":
        int(
            smoke.get(
                "model",
                {},
            ).get(
                "parameter_count",
                -1,
            )
        )
        == 9_603,

    "model_parameters_changed":
        float(
            smoke.get(
                "model",
                {},
            ).get(
                "parameter_change_sum",
                0.0,
            )
        )
        > 0.0,

    "all_numeric_outputs_finite":
        all(
            value is not None
            and math.isfinite(
                float(value)
            )
            for value in numeric_values
        ),

    "feature_alignment_passed":
        feature_alignment.get(
            "all_checks_passed"
        )
        is True,

    "scientific_protocol_locked":
        protocol.get("status")
        == "locked"
        and protocol.get(
            "all_checks_passed"
        )
        is True,
}


all_checks_passed = all(
    validation_checks.values()
)


locked_summary = {
    "protocol_version":
        "early_lodo_smoke_lock_v2_1",

    "status":
        (
            "locked"
            if all_checks_passed
            else "failed"
        ),

    "locked_at":
        utc_now(),

    "scope": {
        "purpose":
            "Pipeline smoke test only",

        "scientific_result":
            False,

        "held_out_device":
            smoke[
                "held_out_device"
            ],

        "model":
            smoke[
                "model_name"
            ],

        "seed":
            smoke["seed"],

        "epoch_count":
            smoke[
                "epoch_count"
            ],
    },

    "sample_sizes":
        sample_sizes,

    "overlap_counts":
        overlap_counts,

    "model":
        smoke["model"],

    "losses":
        losses,

    "validation_macro_f1":
        validation_metrics[
            "macro_f1_present_classes"
        ],

    "test_macro_f1":
        test_metrics[
            "macro_f1_present_classes"
        ],

    "interpretation": (
        "The smoke run validates data loading, "
        "strict LODO membership filtering, "
        "fold-only scaler fitting, model creation, "
        "one-epoch optimization and metric output. "
        "Its metrics must not be included as "
        "scientific experimental results."
    ),

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json(
    LOCKED_SUMMARY,
    locked_summary,
)


completion_text = f"""# Faz 2F — Erken LODO Smoke Testi Tamamlandı

Kilitleme zamanı: `{locked_summary["locked_at"]}`

## Smoke testi

- Tutulan cihaz: `Danmini_Doorbell`
- Model: `tinyml_mlp`
- Seed: `2026`
- Epoch: `1`
- Eğitim örneği: `{sample_sizes["train"]}`
- Doğrulama örneği: `{sample_sizes["validation"]}`
- Test örneği: `{sample_sizes["test"]}`
- Model parametresi: `{smoke["model"]["parameter_count"]}`

## Örtüşme denetimi

- Eğitim–test örtüşmesi: `0`
- Doğrulama–test örtüşmesi: `0`
- Eğitim–doğrulama örtüşmesi: `0`

## Sonuç

Veri okuma, cihaz üyeliği, strict LODO filtreleme, yalnız eğitim
verisiyle scaler öğrenme, model oluşturma, optimizasyon ve metrik
hesaplama zinciri başarıyla doğrulandı.

Smoke testinde üretilen Macro-F1 ve diğer performans değerleri bilimsel
deney sonucu değildir ve makalede performans sonucu olarak kullanılamaz.

## Dosyalar

- `{SMOKE_JSON.relative_to(PROJECT_ROOT)}`
- `{SMOKE_REPORT.relative_to(PROJECT_ROOT)}`
- `{LOCKED_SUMMARY.relative_to(PROJECT_ROOT)}`
- `{MANIFEST_FILE.relative_to(PROJECT_ROOT)}`
"""


COMPLETION_NOTE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_NOTE.write_text(
    completion_text,
    encoding="utf-8",
)


manifest_paths = [
    SMOKE_SCRIPT,
    SMOKE_JSON,
    SMOKE_REPORT,
    FEATURE_ALIGNMENT_SUMMARY,
    PROTOCOL_FILE,
    LOCKED_SUMMARY,
    COMPLETION_NOTE,
]

manifest_rows = [
    {
        "relative_path":
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    }
    for path in manifest_paths
]


write_csv(
    MANIFEST_FILE,
    manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


print("=" * 78)
print("PHASE 2F EARLY LODO SMOKE TEST LOCK")
print("=" * 78)
print(
    f"Held-out device : "
    f"{smoke['held_out_device']}"
)
print(
    f"Model           : "
    f"{smoke['model_name']}"
)
print(
    f"Seed            : "
    f"{smoke['seed']}"
)
print(
    f"Epoch count     : "
    f"{smoke['epoch_count']}"
)
print(
    f"Train samples   : "
    f"{sample_sizes['train']:,}"
)
print(
    f"Validation      : "
    f"{sample_sizes['validation']:,}"
)
print(
    f"Test samples    : "
    f"{sample_sizes['test']:,}"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Locked summary : {LOCKED_SUMMARY}")
print(f"Manifest       : {MANIFEST_FILE}")
print(f"Completion note: {COMPLETION_NOTE}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2F EARLY LODO "
        "SMOKE TEST LOCK FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2F EARLY LODO "
    "SMOKE TEST LOCKED"
)
