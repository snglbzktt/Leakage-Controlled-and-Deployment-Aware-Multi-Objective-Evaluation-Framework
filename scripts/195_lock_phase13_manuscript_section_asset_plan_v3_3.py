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
CONFIG = ROOT / "configs" / "protocols"
ASSET_ROOT = ROOT / "results" / "v2" / "manuscript_assets_v3_3"

CLOSURE_PATH = AUDIT / "phase13_manuscript_assets_closure_v3_3.json"
MANIFEST_PATH = AUDIT / "phase13_manuscript_assets_manifest_v3_3.json"

PLAN_JSON = CONFIG / "phase13_manuscript_section_asset_plan_v3_3.json"
PLAN_CSV = AUDIT / "phase13_manuscript_section_asset_plan_v3_3.csv"
PLAN_MD = AUDIT / "phase13_manuscript_section_asset_plan_v3_3.md"
LOCK_JSON = AUDIT / "phase13_manuscript_section_asset_plan_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None

    for attempt in range(1, WINDOWS_FILE_RETRY_COUNT + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            last_error = error
            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break
            time.sleep(WINDOWS_FILE_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: {destination}"
    ) from last_error


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    replace_with_retry(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    replace_with_retry(temporary, path)


for required_path in (CLOSURE_PATH, MANIFEST_PATH):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for output_path in (PLAN_JSON, PLAN_CSV, PLAN_MD, LOCK_JSON):
    if output_path.exists():
        raise FileExistsError(
            "Manuscript section/asset plan already exists; refusing to overwrite: "
            f"{output_path}"
        )

closure = read_json(CLOSURE_PATH)
manifest = read_json(MANIFEST_PATH)

entry_checks = {
    "phase13_assets_locked": closure.get("phase13_manuscript_assets_locked") is True,
    "closure_ready_for_drafting": closure.get("ready_for_manuscript_drafting") is True,
    "closure_all_checks_passed": closure.get("all_checks_passed") is True,
    "closure_asset_count_12": closure.get("asset_count") == 12,
    "closure_figure_count_5": closure.get("figure_count") == 5,
    "closure_table_count_7": closure.get("table_count") == 7,
    "manifest_asset_count_12": manifest.get("asset_count") == 12,
    "manifest_ready_for_drafting": manifest.get("ready_for_manuscript_drafting") is True,
    "manifest_all_checks_passed": manifest.get("all_checks_passed") is True,
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Manuscript planning entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

manifest_assets = {
    asset["asset_id"]: asset
    for asset in manifest["assets"]
}

expected_assets = {
    "figure_01_methodology_overview",
    "figure_02_lodo_generalization",
    "figure_03_compression_tradeoff",
    "figure_04_pareto_selection",
    "figure_05_input_robustness",
    "table_01_dataset_protocol",
    "table_02_classical_baselines",
    "table_03_compression_results",
    "table_04_deployment_benchmarks",
    "table_05_statistical_analysis",
    "table_06_robustness_results",
    "table_07_reproducibility_release",
}

if set(manifest_assets) != expected_assets:
    raise RuntimeError(
        "Manifest asset set does not match the locked 12-asset plan."
    )

sections = [
    {
        "order": 1,
        "section_id": "I",
        "section_title": "Introduction",
        "purpose": (
            "Define the IoT intrusion-detection problem, the TinyML deployment "
            "constraint, the research gap, and the paper contributions."
        ),
        "primary_assets": [],
        "writing_rule": (
            "No experimental result should be introduced here beyond high-level "
            "motivation and contribution statements."
        ),
    },
    {
        "order": 2,
        "section_id": "II",
        "section_title": "Related Work",
        "purpose": (
            "Position the work against IoT IDS, N-BaIoT evaluation, TinyML, "
            "compression/quantization, device-generalization, and robustness studies."
        ),
        "primary_assets": [],
        "writing_rule": (
            "External literature belongs here; locked project results must not be "
            "used as substitutes for citations."
        ),
    },
    {
        "order": 3,
        "section_id": "III",
        "section_title": "Methodology and Evaluation Protocol",
        "purpose": (
            "Describe the locked end-to-end workflow, N-BaIoT study protocol, "
            "strict LODO design, compression matrix, and test-isolation rules."
        ),
        "primary_assets": [
            "figure_01_methodology_overview",
            "table_01_dataset_protocol",
        ],
        "writing_rule": (
            "Report only protocol facts supported by the locked Phase 3–6 evidence. "
            "Do not claim final model superiority in this section."
        ),
    },
    {
        "order": 4,
        "section_id": "IV",
        "section_title": "Experimental Results",
        "purpose": (
            "Present generalization, baseline, compression, deployment, statistical, "
            "selection, and robustness results in the locked experimental order."
        ),
        "primary_assets": [
            "figure_02_lodo_generalization",
            "table_02_classical_baselines",
            "figure_03_compression_tradeoff",
            "table_03_compression_results",
            "table_04_deployment_benchmarks",
            "table_05_statistical_analysis",
            "figure_04_pareto_selection",
            "figure_05_input_robustness",
            "table_06_robustness_results",
        ],
        "writing_rule": (
            "Preserve the distinction between descriptive rankings, inferential "
            "results, predeclared selection, and post-selection robustness."
        ),
    },
    {
        "order": 5,
        "section_id": "V",
        "section_title": "Reproducibility and Release Integrity",
        "purpose": (
            "Document the selected family, canonical seed, artifact checks, "
            "deterministic release hashes, and release-content exclusions."
        ),
        "primary_assets": [
            "table_07_reproducibility_release",
        ],
        "writing_rule": (
            "Use exact locked hashes and integrity statements. Do not present "
            "the release bundle as containing raw or processed datasets."
        ),
    },
    {
        "order": 6,
        "section_id": "VI",
        "section_title": "Discussion",
        "purpose": (
            "Interpret the combined evidence: strong strict-LODO generalization, "
            "compression/deployment trade-offs, statistical limitations, selected "
            "P50-QAT family, and perturbation sensitivity."
        ),
        "primary_assets": [],
        "writing_rule": (
            "Interpret rather than repeat tables. State that zero Holm post-hoc "
            "rejections do not establish equivalence."
        ),
    },
    {
        "order": 7,
        "section_id": "VII",
        "section_title": "Limitations and Threats to Validity",
        "purpose": (
            "State dataset scope, five-seed inferential resolution, hardware-specific "
            "CPU benchmarking, two-class held-out devices, and the deferred optional "
            "external-dataset validation."
        ),
        "primary_assets": [],
        "writing_rule": (
            "Phase 12 must be described as deferred/optional and must not be implied "
            "to have been completed."
        ),
    },
    {
        "order": 8,
        "section_id": "VIII",
        "section_title": "Conclusion",
        "purpose": (
            "Summarize the supported contribution without adding new metrics, "
            "experiments, or claims."
        ),
        "primary_assets": [],
        "writing_rule": (
            "No new numerical claim may appear here unless already established "
            "in the locked evidence."
        ),
    },
]

result_subsections = [
    {
        "subsection_id": "IV-A",
        "title": "Strict Leave-One-Device-Out Generalization",
        "assets": ["figure_02_lodo_generalization"],
        "core_message": (
            "Report the 27-run 9-device × 3-model LODO matrix and the absence of "
            "severe LODO cells under the locked thresholds."
        ),
    },
    {
        "subsection_id": "IV-B",
        "title": "Classical Baseline Performance",
        "assets": ["table_02_classical_baselines"],
        "core_message": (
            "Present the four classical baselines as descriptive comparisons only; "
            "the ranking is not final model selection."
        ),
    },
    {
        "subsection_id": "IV-C",
        "title": "Compression Accuracy Trade-offs",
        "assets": [
            "figure_03_compression_tradeoff",
            "table_03_compression_results",
        ],
        "core_message": (
            "Present all 22 five-seed architecture–variant groups and identify the "
            "three Phase 5 fragility groups without treating Phase 5 as selection."
        ),
    },
    {
        "subsection_id": "IV-D",
        "title": "CPU Deployment Benchmarks",
        "assets": ["table_04_deployment_benchmarks"],
        "core_message": (
            "Report isolated one-thread CPU latency, throughput, state size, and "
            "memory measurements without validation/test-data access."
        ),
    },
    {
        "subsection_id": "IV-E",
        "title": "Statistical Analysis",
        "assets": ["table_05_statistical_analysis"],
        "core_message": (
            "Report 16 nominal omnibus rejections, zero Holm-corrected post-hoc "
            "rejections across 160 comparisons, and the n=5 exact-p limitation."
        ),
    },
    {
        "subsection_id": "IV-F",
        "title": "Multi-objective Model-Family Selection",
        "assets": ["figure_04_pareto_selection"],
        "core_message": (
            "Explain the predeclared eligibility → Pareto → deployment ordering. "
            "Balanced/relaxed select TinyML-MLP P50-QAT; strict sensitivity selects "
            "TinyML-MLP P50-FP32-FT."
        ),
    },
    {
        "subsection_id": "IV-G",
        "title": "Input Robustness",
        "assets": [
            "figure_05_input_robustness",
            "table_06_robustness_results",
        ],
        "core_message": (
            "Report 27 locked evaluations, five severe-fragility evaluations all "
            "for HGB, and nine material-sensitivity evaluations; no retraining or "
            "reselection occurred."
        ),
    },
]

placements: list[dict[str, Any]] = []

for section in sections:
    for position, asset_id in enumerate(section["primary_assets"], start=1):
        placements.append(
            {
                "section_order": section["order"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "asset_position_in_section": position,
                "asset_id": asset_id,
                "asset_type": manifest_assets[asset_id]["asset_type"],
                "primary_placement": True,
            }
        )

placed_assets = [row["asset_id"] for row in placements]

placement_checks = {
    "all_12_assets_have_primary_placement": set(placed_assets) == expected_assets,
    "each_asset_placed_exactly_once": all(
        placed_assets.count(asset_id) == 1
        for asset_id in expected_assets
    ),
    "methodology_contains_figure01": (
        "figure_01_methodology_overview"
        in sections[2]["primary_assets"]
    ),
    "methodology_contains_table01": (
        "table_01_dataset_protocol"
        in sections[2]["primary_assets"]
    ),
    "results_contains_9_assets": len(sections[3]["primary_assets"]) == 9,
    "reproducibility_contains_table07": (
        sections[4]["primary_assets"]
        == ["table_07_reproducibility_release"]
    ),
    "phase12_deferred_optional": (
        closure.get("phase12_external_dataset") == "deferred_optional"
    ),
}

failed_placement_checks = [
    name for name, passed in placement_checks.items() if not passed
]

if failed_placement_checks:
    raise RuntimeError(
        "Manuscript placement validation failed: "
        + ", ".join(failed_placement_checks)
    )

writing_guardrails = [
    "Do not use test results for retroactive model selection.",
    "Do not describe Phase 4 classical-baseline ranking as final model selection.",
    "Do not describe Phase 5 compression results as final model selection.",
    "Do not interpret zero Holm-corrected post-hoc rejections as model equivalence.",
    "Do not claim external-dataset validation was completed; Phase 12 is deferred/optional.",
    "Do not claim robustness experiments retrained or changed the selected model.",
    "Do not state that the release archive contains raw data, processed data, or nonselected checkpoints.",
    "Use the exact locked model-family label tinyml_mlp::P50-QAT and canonical seed 2026 when reproducibility details are required.",
    "Use locked tables/figures as the source of manuscript numbers rather than recomputing metrics.",
]

plan = {
    "status": "locked",
    "phase": 13,
    "artifact_name": "manuscript_section_asset_plan",
    "protocol_version": "phase13_manuscript_structure_v3_3",
    "locked_at_utc": utc_now(),
    "target_venue_format": "IEEE-style journal manuscript structure",
    "section_count": len(sections),
    "result_subsection_count": len(result_subsections),
    "asset_count": len(expected_assets),
    "sections": sections,
    "result_subsections": result_subsections,
    "asset_placements": placements,
    "writing_guardrails": writing_guardrails,
    "source_closure": {
        "path": str(CLOSURE_PATH),
        "sha256": sha256_file(CLOSURE_PATH),
    },
    "source_manifest": {
        "path": str(MANIFEST_PATH),
        "sha256": sha256_file(MANIFEST_PATH),
    },
    "entry_checks": entry_checks,
    "placement_checks": placement_checks,
    "phase12_external_dataset": "deferred_optional",
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript_drafting": True,
    "all_checks_passed": True,
}

atomic_json(PLAN_JSON, plan)

PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)

with PLAN_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "section_order",
            "section_id",
            "section_title",
            "asset_position_in_section",
            "asset_id",
            "asset_type",
            "primary_placement",
        ],
    )
    writer.writeheader()
    writer.writerows(placements)

md_lines = [
    "# Phase 13 Manuscript Section and Asset Plan",
    "",
    "## Main sections",
    "",
]

for section in sections:
    md_lines.append(
        f"### {section['section_id']}. {section['section_title']}"
    )
    md_lines.append("")
    md_lines.append(section["purpose"])
    md_lines.append("")
    md_lines.append(
        "**Primary assets:** "
        + (
            ", ".join(section["primary_assets"])
            if section["primary_assets"]
            else "None"
        )
    )
    md_lines.append("")
    md_lines.append(
        f"**Writing rule:** {section['writing_rule']}"
    )
    md_lines.append("")

md_lines.extend(
    [
        "## Experimental Results subsections",
        "",
    ]
)

for subsection in result_subsections:
    md_lines.append(
        f"### {subsection['subsection_id']} — {subsection['title']}"
    )
    md_lines.append("")
    md_lines.append(
        "**Assets:** " + ", ".join(subsection["assets"])
    )
    md_lines.append("")
    md_lines.append(subsection["core_message"])
    md_lines.append("")

md_lines.extend(
    [
        "## Locked writing guardrails",
        "",
    ]
)

for guardrail in writing_guardrails:
    md_lines.append(f"- {guardrail}")

md_lines.append("")

atomic_text(PLAN_MD, "\n".join(md_lines))

lock = {
    "status": "locked",
    "phase": 13,
    "artifact_name": "manuscript_section_asset_plan",
    "locked_at_utc": utc_now(),
    "plan_json": {
        "path": str(PLAN_JSON),
        "sha256": sha256_file(PLAN_JSON),
    },
    "plan_csv": {
        "path": str(PLAN_CSV),
        "sha256": sha256_file(PLAN_CSV),
    },
    "plan_markdown": {
        "path": str(PLAN_MD),
        "sha256": sha256_file(PLAN_MD),
    },
    "section_count": 8,
    "result_subsection_count": 7,
    "asset_count": 12,
    "all_12_assets_placed_exactly_once": True,
    "phase12_external_dataset": "deferred_optional",
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript_drafting": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 100)
print("PHASE 13 MANUSCRIPT SECTION / ASSET PLAN")
print("=" * 100)
print(f"Main sections                    : {len(sections)}")
print(f"Results subsections              : {len(result_subsections)}")
print(f"Locked manuscript assets         : {len(expected_assets)}")
print("All assets placed exactly once   : True")
print("Phase 12 external dataset        : deferred_optional")
print()
for section in sections:
    asset_text = (
        ", ".join(section["primary_assets"])
        if section["primary_assets"]
        else "-"
    )
    print(
        f"{section['section_id']:<4} "
        f"{section['section_title']:<42} | "
        f"assets={asset_text}"
    )
print()
print("Experimental Results ordering:")
for subsection in result_subsections:
    print(
        f"{subsection['subsection_id']:<5} "
        f"{subsection['title']:<42} | "
        f"assets={', '.join(subsection['assets'])}"
    )
print()
print(f"Plan JSON                        : {PLAN_JSON}")
print(f"Plan CSV                         : {PLAN_CSV}")
print(f"Plan Markdown                    : {PLAN_MD}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model inference performed        : False")
print("Test arrays accessed             : False")
print("Source files mutated             : False")
print("Ready for manuscript drafting   : True")
print("All checks passed               : True")
print("PHASE 13 MANUSCRIPT STRUCTURE LOCKED")
