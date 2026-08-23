from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

INVENTORY_PATH = (
    AUDIT
    / "phase13_manuscript_evidence_inventory_v3_3.json"
)

PHASE11_LOCK_PATH = (
    AUDIT
    / "phase11_input_robustness_locked_v3_3.json"
)

OUTPUT_PLAN_JSON = (
    ROOT
    / "configs"
    / "protocols"
    / "phase13_manuscript_asset_plan_v3_3.json"
)

OUTPUT_PLAN_CSV = (
    AUDIT
    / "phase13_manuscript_asset_plan_v3_3.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase13_manuscript_asset_plan_preflight_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase13_manuscript_asset_plan_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase13_manuscript_asset_plan_lock_manifest_v3_3.json"
)

PROTOCOL_VERSION = (
    "phase13_manuscript_asset_plan_v3_3"
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


ASSET_SPECS = [
    {
        "asset_id": "figure_01_methodology_overview",
        "asset_type": "figure",
        "title": (
            "Leakage-Controlled TinyML IDS "
            "Experimental Pipeline"
        ),
        "source_sections": [
            "phase3_lodo",
            "phase5_compression",
            "phase6_deployment_benchmarks",
            "phase8_selection",
            "phase9_deployment_artifact",
            "phase10_release",
            "phase11_robustness",
        ],
        "visual_form": (
            "vector workflow diagram"
        ),
        "scientific_purpose": (
            "Summarize data controls, model training, "
            "compression, deployment benchmarking, "
            "selection, packaging, and robustness."
        ),
        "primary_message": (
            "The study is an end-to-end, leakage-aware "
            "and deployment-oriented TinyML IDS workflow."
        ),
        "planned_outputs": [
            "SVG",
            "PDF",
            "PNG",
        ],
        "data_access": (
            "protocol and lock metadata only"
        ),
    },
    {
        "asset_id": "figure_02_lodo_generalization",
        "asset_type": "figure",
        "title": (
            "Leave-One-Device-Out Generalization"
        ),
        "source_sections": [
            "phase3_lodo",
        ],
        "visual_form": (
            "device-level grouped point or bar chart"
        ),
        "scientific_purpose": (
            "Show unseen-device generalization across "
            "TinyML-MLP, Compact-DNN, and HGB."
        ),
        "primary_message": (
            "Neural TinyML models retain strong "
            "performance under strict device holdout."
        ),
        "planned_outputs": [
            "PDF",
            "PNG",
            "CSV",
        ],
        "data_access": (
            "locked Phase 3 summaries only"
        ),
    },
    {
        "asset_id": "figure_03_compression_tradeoff",
        "asset_type": "figure",
        "title": (
            "Compression Accuracy and Resource Trade-off"
        ),
        "source_sections": [
            "phase5_compression",
            "phase6_deployment_benchmarks",
        ],
        "visual_form": (
            "Pareto scatter with model size or latency "
            "against Macro-F1"
        ),
        "scientific_purpose": (
            "Compare B0, pruning, DQ, PTQ, QAT, "
            "P25-QAT, and P50-QAT."
        ),
        "primary_message": (
            "P50-QAT provides a favorable TinyML "
            "deployment trade-off."
        ),
        "planned_outputs": [
            "PDF",
            "PNG",
            "CSV",
        ],
        "data_access": (
            "locked aggregate result files only"
        ),
    },
    {
        "asset_id": "figure_04_pareto_selection",
        "asset_type": "figure",
        "title": (
            "Final Model-Family Pareto Selection"
        ),
        "source_sections": [
            "phase6_deployment_benchmarks",
            "phase7_statistics",
            "phase8_selection",
        ],
        "visual_form": (
            "annotated Pareto frontier"
        ),
        "scientific_purpose": (
            "Explain why TinyML-MLP P50-QAT was "
            "selected under the balanced policy."
        ),
        "primary_message": (
            "The final family was chosen from "
            "multi-objective deployment evidence, "
            "not test-metric ranking of checkpoints."
        ),
        "planned_outputs": [
            "PDF",
            "PNG",
            "CSV",
        ],
        "data_access": (
            "locked Phase 6-8 summaries only"
        ),
    },
    {
        "asset_id": "figure_05_input_robustness",
        "asset_type": "figure",
        "title": (
            "Input Perturbation Robustness"
        ),
        "source_sections": [
            "phase11_robustness",
        ],
        "visual_form": (
            "condition-wise Macro-F1 line chart"
        ),
        "scientific_purpose": (
            "Compare TinyML B0, P50-QAT, and HGB "
            "under noise, masking, and scale drift."
        ),
        "primary_message": (
            "TinyML models are substantially more "
            "stable than HGB under noise and "
            "scale drift."
        ),
        "planned_outputs": [
            "PDF",
            "PNG",
            "CSV",
        ],
        "data_access": (
            "locked Phase 11 aggregate metrics only"
        ),
    },
    {
        "asset_id": "table_01_dataset_protocol",
        "asset_type": "table",
        "title": (
            "Dataset, Split, Leakage-Control, and "
            "Evaluation Protocol"
        ),
        "source_sections": [
            "phase3_lodo",
            "phase4_classical_baselines",
            "phase5_compression",
        ],
        "visual_form": (
            "manuscript summary table"
        ),
        "scientific_purpose": (
            "Document sample counts, feature count, "
            "class order, split controls, and test-once "
            "policy."
        ),
        "primary_message": (
            "Evaluation follows locked, leakage-aware "
            "and test-isolated protocols."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked manifests and summaries only"
        ),
    },
    {
        "asset_id": "table_02_classical_baselines",
        "asset_type": "table",
        "title": (
            "Classical Baseline Performance"
        ),
        "source_sections": [
            "phase4_classical_baselines",
        ],
        "visual_form": (
            "performance comparison table"
        ),
        "scientific_purpose": (
            "Report LR, DT, RF, and HGB results."
        ),
        "primary_message": (
            "HGB is the strongest clean-data classical "
            "baseline but is not the most robust."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked Phase 4 summaries only"
        ),
    },
    {
        "asset_id": "table_03_compression_results",
        "asset_type": "table",
        "title": (
            "Five-Seed Compression Results"
        ),
        "source_sections": [
            "phase5_compression",
        ],
        "visual_form": (
            "mean, standard deviation, weighted "
            "Macro-F1, size, and fragility table"
        ),
        "scientific_purpose": (
            "Summarize all 22 architecture-variant "
            "groups."
        ),
        "primary_message": (
            "QAT preserves accuracy better than "
            "post-training dynamic quantization."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked Phase 5 aggregate files only"
        ),
    },
    {
        "asset_id": "table_04_deployment_benchmarks",
        "asset_type": "table",
        "title": (
            "CPU Deployment Benchmark Results"
        ),
        "source_sections": [
            "phase6_deployment_benchmarks",
        ],
        "visual_form": (
            "latency, throughput, size, parameter, "
            "and MAC comparison table"
        ),
        "scientific_purpose": (
            "Present deployment-oriented efficiency."
        ),
        "primary_message": (
            "The selected static-INT8 family reduces "
            "resource demand while retaining accuracy."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked Phase 6 benchmark summaries only"
        ),
    },
    {
        "asset_id": "table_05_statistical_analysis",
        "asset_type": "table",
        "title": (
            "Omnibus and Post-hoc Statistical Results"
        ),
        "source_sections": [
            "phase7_statistics",
        ],
        "visual_form": (
            "confirmatory and exploratory test table"
        ),
        "scientific_purpose": (
            "Report permutation omnibus results and "
            "Holm-adjusted pairwise findings."
        ),
        "primary_message": (
            "Global differences are detectable, while "
            "five-seed pairwise resolution remains "
            "limited."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked Phase 7 statistical outputs only"
        ),
    },
    {
        "asset_id": "table_06_robustness_results",
        "asset_type": "table",
        "title": (
            "Input Robustness Results"
        ),
        "source_sections": [
            "phase11_robustness",
        ],
        "visual_form": (
            "27-run robustness result table"
        ),
        "scientific_purpose": (
            "Report clean and perturbed performance, "
            "FNR, degradation, and fragility."
        ),
        "primary_message": (
            "HGB has five severe fragility cases; "
            "TinyML models have none."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "locked Phase 11 result files only"
        ),
    },
    {
        "asset_id": "table_07_reproducibility_release",
        "asset_type": "table",
        "title": (
            "Reproducibility and Release Artifacts"
        ),
        "source_sections": [
            "phase8_selection",
            "phase9_deployment_artifact",
            "phase10_release",
            "phase11_robustness",
        ],
        "visual_form": (
            "artifact and integrity table"
        ),
        "scientific_purpose": (
            "Document selected family, canonical "
            "checkpoint, artifact representation, "
            "release hashes, and lock status."
        ),
        "primary_message": (
            "The selected deployment package is "
            "independently verified and reproducibly "
            "locked."
        ),
        "planned_outputs": [
            "CSV",
            "DOCX-ready",
            "LaTeX-ready",
        ],
        "data_access": (
            "external lock and verification files only"
        ),
    },
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


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
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(
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
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    replace_with_retry(
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


for path in (
    INVENTORY_PATH,
    PHASE11_LOCK_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PLAN_JSON,
    OUTPUT_PLAN_CSV,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 asset-plan output already "
            f"exists; refusing to overwrite: {output_path}"
        )

inventory = read_json(
    INVENTORY_PATH
)

phase11_lock = read_json(
    PHASE11_LOCK_PATH
)

entry_checks = {
    "inventory_completed": (
        inventory.get("status")
        == "completed"
        and inventory.get(
            "all_checks_passed"
        )
        is True
    ),
    "inventory_ready_for_asset_plan": (
        inventory.get(
            "ready_for_asset_plan"
        )
        is True
    ),
    "inventory_missing_sections_zero": (
        inventory.get(
            "missing_sections"
        )
        == []
    ),
    "phase11_locked": (
        phase11_lock.get("status")
        == "locked"
        and phase11_lock.get(
            "phase11_locked"
        )
        is True
        and phase11_lock.get(
            "ready_for_figures_and_reporting"
        )
        is True
        and phase11_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase12_deferred_optional": (
        inventory.get(
            "phase12_external_dataset"
        )
        == "deferred_optional"
    ),
    "asset_count_12": (
        len(ASSET_SPECS) == 12
    ),
    "figure_count_5": (
        sum(
            1
            for asset in ASSET_SPECS
            if asset["asset_type"]
            == "figure"
        )
        == 5
    ),
    "table_count_7": (
        sum(
            1
            for asset in ASSET_SPECS
            if asset["asset_type"]
            == "table"
        )
        == 7
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
        "Phase 13 asset-plan entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

inventory_files = list(
    inventory["files"]
)

files_by_section: dict[
    str,
    list[dict[str, Any]],
] = {}

for record in inventory_files:
    files_by_section.setdefault(
        record["section"],
        [],
    ).append(record)

all_available_sections = set(
    files_by_section.keys()
)

asset_rows: list[
    dict[str, Any]
] = []

resolved_assets: list[
    dict[str, Any]
] = []

for order, asset in enumerate(
    ASSET_SPECS,
    start=1,
):
    required_sections = set(
        asset["source_sections"]
    )

    missing_sections = sorted(
        required_sections
        - all_available_sections
    )

    if missing_sections:
        raise RuntimeError(
            f"{asset['asset_id']} missing evidence "
            "sections: "
            + ", ".join(missing_sections)
        )

    evidence_records = []

    for section in asset[
        "source_sections"
    ]:
        evidence_records.extend(
            files_by_section[section]
        )

    evidence_records = sorted(
        evidence_records,
        key=lambda record: (
            record["section"],
            record["relative_path"],
        ),
    )

    resolved = {
        **asset,
        "asset_order": order,
        "evidence_file_count": len(
            evidence_records
        ),
        "evidence_files": [
            {
                "section": record[
                    "section"
                ],
                "relative_path": record[
                    "relative_path"
                ],
                "sha256": record[
                    "sha256"
                ],
            }
            for record in evidence_records
        ],
        "generation_status": "planned",
        "model_inference_required": False,
        "test_array_access_required": False,
        "source_mutation_allowed": False,
    }

    resolved_assets.append(
        resolved
    )

    asset_rows.append(
        {
            "asset_order": order,
            "asset_id": asset[
                "asset_id"
            ],
            "asset_type": asset[
                "asset_type"
            ],
            "title": asset[
                "title"
            ],
            "source_sections": (
                ";".join(
                    asset[
                        "source_sections"
                    ]
                )
            ),
            "evidence_file_count": len(
                evidence_records
            ),
            "visual_form": asset[
                "visual_form"
            ],
            "primary_message": asset[
                "primary_message"
            ],
            "planned_outputs": (
                ";".join(
                    asset[
                        "planned_outputs"
                    ]
                )
            ),
            "generation_status": "planned",
            "model_inference_required": (
                False
            ),
            "test_array_access_required": (
                False
            ),
        }
    )

plan = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "manuscript_asset_plan"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "phase12_external_dataset": (
        "deferred_optional"
    ),
    "scope": {
        "figure_count": 5,
        "table_count": 7,
        "total_asset_count": 12,
        "primary_manuscript_target": (
            "IEEE Access-style article"
        ),
        "assets_based_only_on_locked_evidence": (
            True
        ),
    },
    "scientific_rules": {
        "no_new_model_selection": True,
        "no_checkpoint_reselection": True,
        "no_hyperparameter_tuning": True,
        "no_test_array_access": True,
        "no_model_inference": True,
        "no_source_result_mutation": True,
        "report_descriptive_robustness": True,
        "do_not_claim_cross_dataset_generalization": (
            True
        ),
        "state_external_dataset_as_future_or_optional": (
            True
        ),
    },
    "format_rules": {
        "figure_vector_master_preferred": (
            True
        ),
        "figure_raster_resolution_dpi": 600,
        "figure_font_embedding_required": (
            True
        ),
        "figure_color_blind_readability_required": (
            True
        ),
        "figure_grayscale_readability_required": (
            True
        ),
        "table_machine_readable_csv_required": (
            True
        ),
        "table_docx_ready_required": True,
        "table_latex_ready_required": True,
        "decimal_precision_policy": (
            "preserve locked values in source tables; "
            "round only presentation copies"
        ),
    },
    "assets": resolved_assets,
    "source_inventory": (
        file_record(
            INVENTORY_PATH
        )
    ),
    "phase11_lock": (
        file_record(
            PHASE11_LOCK_PATH
        )
    ),
    "generation_performed": False,
    "model_inference_performed": False,
    "test_array_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PLAN_JSON,
    plan,
)

write_csv(
    OUTPUT_PLAN_CSV,
    asset_rows,
    fieldnames=list(
        asset_rows[0].keys()
    ),
)

preflight_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "asset_ids_unique": (
        len(
            {
                asset["asset_id"]
                for asset in resolved_assets
            }
        )
        == 12
    ),
    "all_assets_have_evidence": all(
        asset[
            "evidence_file_count"
        ]
        > 0
        for asset in resolved_assets
    ),
    "all_assets_planned_only": all(
        asset[
            "generation_status"
        ]
        == "planned"
        for asset in resolved_assets
    ),
    "all_assets_no_inference": all(
        asset[
            "model_inference_required"
        ]
        is False
        for asset in resolved_assets
    ),
    "all_assets_no_test_array_access": (
        all(
            asset[
                "test_array_access_required"
            ]
            is False
            for asset in resolved_assets
        )
    ),
    "generation_not_performed": (
        plan[
            "generation_performed"
        ]
        is False
    ),
    "source_files_not_mutated": (
        plan[
            "source_files_mutated"
        ]
        is False
    ),
}

failed_preflight_checks = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight_checks:
    raise RuntimeError(
        "Phase 13 asset-plan preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 13,
    "artifact_name": (
        "manuscript_asset_plan_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "plan": (
        file_record(
            OUTPUT_PLAN_JSON
        )
    ),
    "plan_csv": (
        file_record(
            OUTPUT_PLAN_CSV
        )
    ),
    "figure_count": 5,
    "table_count": 7,
    "total_asset_count": 12,
    "generation_performed": False,
    "model_inference_performed": False,
    "test_array_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "manuscript_asset_plan"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "plan": (
        file_record(
            OUTPUT_PLAN_JSON
        )
    ),
    "plan_csv": (
        file_record(
            OUTPUT_PLAN_CSV
        )
    ),
    "preflight": (
        file_record(
            OUTPUT_PREFLIGHT
        )
    ),
    "figure_count": 5,
    "table_count": 7,
    "total_asset_count": 12,
    "generation_performed": False,
    "model_inference_performed": False,
    "test_array_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "manuscript_asset_plan"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            INVENTORY_PATH
        ),
        file_record(
            PHASE11_LOCK_PATH
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_PLAN_JSON
        ),
        file_record(
            OUTPUT_PLAN_CSV
        ),
        file_record(
            OUTPUT_PREFLIGHT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "figure_count": 5,
    "table_count": 7,
    "total_asset_count": 12,
    "generation_performed": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 13 MANUSCRIPT ASSET PLAN")
print("=" * 92)
print(
    "Phase 12 external dataset       : DEFERRED / OPTIONAL"
)
print(
    "Evidence inventory ready        : True"
)
print(
    "Figures planned                 : 5"
)
print(
    "Tables planned                  : 7"
)
print(
    "Total manuscript assets         : 12"
)
print()
print("PLANNED FIGURES")
print("-" * 92)

for asset in resolved_assets:
    if asset["asset_type"] == "figure":
        print(
            f"{asset['asset_order']:02d}. "
            f"{asset['asset_id']:<36} | "
            f"evidence={asset['evidence_file_count']}"
        )

print()
print("PLANNED TABLES")
print("-" * 92)

for asset in resolved_assets:
    if asset["asset_type"] == "table":
        print(
            f"{asset['asset_order']:02d}. "
            f"{asset['asset_id']:<36} | "
            f"evidence={asset['evidence_file_count']}"
        )

print()
print("=" * 92)
print("ASSET PLAN PREFLIGHT SUMMARY")
print("=" * 92)
print(
    "Assets based on locked evidence : True"
)
print(
    "New model selection             : False"
)
print(
    "Model inference performed       : False"
)
print(
    "Test arrays accessed            : False"
)
print(
    "Source files mutated            : False"
)
print(
    "Asset generation performed      : False"
)
print(
    "Plan status                     : LOCKED"
)
print(
    "Ready for asset generation      : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 13 MANUSCRIPT ASSET PLAN LOCKED"
)
