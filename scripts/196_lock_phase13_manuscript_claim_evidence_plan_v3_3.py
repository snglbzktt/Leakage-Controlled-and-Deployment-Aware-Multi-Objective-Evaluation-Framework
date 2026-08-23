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

STRUCTURE_PLAN = CONFIG / "phase13_manuscript_section_asset_plan_v3_3.json"
STRUCTURE_LOCK = AUDIT / "phase13_manuscript_section_asset_plan_locked_v3_3.json"
ASSET_MANIFEST = AUDIT / "phase13_manuscript_assets_manifest_v3_3.json"
ASSET_CLOSURE = AUDIT / "phase13_manuscript_assets_closure_v3_3.json"

PLAN_JSON = CONFIG / "phase13_manuscript_claim_evidence_plan_v3_3.json"
PLAN_CSV = AUDIT / "phase13_manuscript_claim_evidence_plan_v3_3.csv"
PLAN_MD = AUDIT / "phase13_manuscript_claim_evidence_plan_v3_3.md"
LOCK_JSON = AUDIT / "phase13_manuscript_claim_evidence_plan_locked_v3_3.json"

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


for required in (
    STRUCTURE_PLAN,
    STRUCTURE_LOCK,
    ASSET_MANIFEST,
    ASSET_CLOSURE,
):
    if not required.exists():
        raise FileNotFoundError(required)

for output in (
    PLAN_JSON,
    PLAN_CSV,
    PLAN_MD,
    LOCK_JSON,
):
    if output.exists():
        raise FileExistsError(
            "Claim/evidence plan already exists; refusing to overwrite: "
            f"{output}"
        )

structure = read_json(STRUCTURE_PLAN)
structure_lock = read_json(STRUCTURE_LOCK)
manifest = read_json(ASSET_MANIFEST)
closure = read_json(ASSET_CLOSURE)

entry_checks = {
    "structure_locked": structure.get("status") == "locked",
    "structure_ready": structure.get("ready_for_manuscript_drafting") is True,
    "structure_all_checks": structure.get("all_checks_passed") is True,
    "structure_lock_hash_matches": (
        structure_lock["plan_json"]["sha256"] == sha256_file(STRUCTURE_PLAN)
    ),
    "assets_locked": closure.get("phase13_manuscript_assets_locked") is True,
    "assets_ready": closure.get("ready_for_manuscript_drafting") is True,
    "assets_all_checks": closure.get("all_checks_passed") is True,
    "asset_count_12": manifest.get("asset_count") == 12,
    "phase12_deferred_optional": (
        closure.get("phase12_external_dataset") == "deferred_optional"
    ),
}

failed_entry = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry:
    raise RuntimeError(
        "Claim/evidence planning entry gate failed: "
        + ", ".join(failed_entry)
    )

asset_ids = {
    asset["asset_id"]
    for asset in manifest["assets"]
}

claims = [
    {
        "claim_id": "A01",
        "manuscript_location": "Abstract",
        "claim_role": "problem_and_scope",
        "claim_text": (
            "The study evaluates a TinyML-oriented IoT intrusion-detection pipeline "
            "under device-generalization, compression, deployment, statistical, "
            "selection, and robustness constraints."
        ),
        "evidence_assets": [
            "figure_01_methodology_overview",
            "table_01_dataset_protocol",
        ],
        "claim_strength": "descriptive",
        "drafting_rule": "Use as scope statement; do not add unsupported benchmark claims.",
    },
    {
        "claim_id": "A02",
        "manuscript_location": "Abstract",
        "claim_role": "generalization_result",
        "claim_text": (
            "Strict leave-one-device-out evaluation covered nine devices, three model "
            "families, and 27 scientific runs without severe LODO cells under the locked gate."
        ),
        "evidence_assets": ["figure_02_lodo_generalization"],
        "claim_strength": "quantitative",
        "drafting_rule": "Use exact locked counts and model means only.",
    },
    {
        "claim_id": "A03",
        "manuscript_location": "Abstract",
        "claim_role": "selection_result",
        "claim_text": (
            "The balanced and relaxed multi-objective profiles selected the "
            "TinyML-MLP P50-QAT family; the strict sensitivity profile selected "
            "TinyML-MLP P50-FP32-FT."
        ),
        "evidence_assets": ["figure_04_pareto_selection"],
        "claim_strength": "quantitative",
        "drafting_rule": "Do not describe the strict sensitivity result as the final family.",
    },
    {
        "claim_id": "A04",
        "manuscript_location": "Abstract",
        "claim_role": "robustness_result",
        "claim_text": (
            "Across 27 locked robustness evaluations, five severe-fragility evaluations "
            "occurred and all were associated with HGB; the locked final model was unchanged."
        ),
        "evidence_assets": [
            "figure_05_input_robustness",
            "table_06_robustness_results",
        ],
        "claim_strength": "quantitative",
        "drafting_rule": "Do not imply retraining or post-test reselection.",
    },
    {
        "claim_id": "I01",
        "manuscript_location": "I. Introduction",
        "claim_role": "contribution",
        "claim_text": (
            "Contribution 1: a leakage-aware strict device-generalization evaluation "
            "is performed before compression/model-family selection."
        ),
        "evidence_assets": [
            "figure_01_methodology_overview",
            "figure_02_lodo_generalization",
        ],
        "claim_strength": "descriptive",
        "drafting_rule": "Present as a study-design contribution, not a universal novelty claim.",
    },
    {
        "claim_id": "I02",
        "manuscript_location": "I. Introduction",
        "claim_role": "contribution",
        "claim_text": (
            "Contribution 2: compression accuracy and deployment cost are evaluated "
            "jointly across a locked multi-seed architecture–variant matrix."
        ),
        "evidence_assets": [
            "figure_03_compression_tradeoff",
            "table_03_compression_results",
            "table_04_deployment_benchmarks",
        ],
        "claim_strength": "descriptive",
        "drafting_rule": "Avoid claiming superiority before the selection subsection.",
    },
    {
        "claim_id": "I03",
        "manuscript_location": "I. Introduction",
        "claim_role": "contribution",
        "claim_text": (
            "Contribution 3: model-family choice follows a predeclared eligibility, "
            "Pareto, and deployment-ordering process rather than a weighted score."
        ),
        "evidence_assets": ["figure_04_pareto_selection"],
        "claim_strength": "descriptive",
        "drafting_rule": "State that pairwise p-values were not used as a selection gate.",
    },
    {
        "claim_id": "I04",
        "manuscript_location": "I. Introduction",
        "claim_role": "contribution",
        "claim_text": (
            "Contribution 4: post-selection robustness and deterministic release "
            "integrity are checked without changing the selected model."
        ),
        "evidence_assets": [
            "table_06_robustness_results",
            "table_07_reproducibility_release",
        ],
        "claim_strength": "descriptive",
        "drafting_rule": "Do not imply that the release contains raw or processed datasets.",
    },
    {
        "claim_id": "III01",
        "manuscript_location": "III. Methodology and Evaluation Protocol",
        "claim_role": "workflow",
        "claim_text": (
            "The locked workflow proceeds from strict LODO generalization through "
            "compression, CPU benchmarking, multi-objective selection, artifact "
            "locking, deterministic release, and robustness evaluation."
        ),
        "evidence_assets": ["figure_01_methodology_overview"],
        "claim_strength": "descriptive",
        "drafting_rule": "Preserve experimental ordering exactly.",
    },
    {
        "claim_id": "III02",
        "manuscript_location": "III. Methodology and Evaluation Protocol",
        "claim_role": "protocol",
        "claim_text": (
            "The protocol uses nine held-out devices, three LODO model families, "
            "four classical baselines, two neural architectures, eleven compression "
            "variants, and five locked seeds for the compression matrix."
        ),
        "evidence_assets": ["table_01_dataset_protocol"],
        "claim_strength": "quantitative",
        "drafting_rule": "Use exact locked values only.",
    },
    {
        "claim_id": "IVA01",
        "manuscript_location": "IV-A. Strict Leave-One-Device-Out Generalization",
        "claim_role": "result",
        "claim_text": (
            "The complete strict LODO matrix contains 27 device–model evaluations, "
            "with no severe LODO cells under the locked fragility thresholds."
        ),
        "evidence_assets": ["figure_02_lodo_generalization"],
        "claim_strength": "quantitative",
        "drafting_rule": "Mention the two-class held-out-device caveat where relevant.",
    },
    {
        "claim_id": "IVB01",
        "manuscript_location": "IV-B. Classical Baseline Performance",
        "claim_role": "result",
        "claim_text": (
            "HistGradientBoosting has the highest descriptive mean Macro-F1 among "
            "the four classical baselines."
        ),
        "evidence_assets": ["table_02_classical_baselines"],
        "claim_strength": "descriptive_quantitative",
        "drafting_rule": "Explicitly state that this ranking is not final model selection.",
    },
    {
        "claim_id": "IVC01",
        "manuscript_location": "IV-C. Compression Accuracy Trade-offs",
        "claim_role": "result",
        "claim_text": (
            "The compression study contains 22 five-seed architecture–variant groups "
            "and exposes distinct accuracy–size trade-offs."
        ),
        "evidence_assets": [
            "figure_03_compression_tradeoff",
            "table_03_compression_results",
        ],
        "claim_strength": "quantitative",
        "drafting_rule": "Keep Phase 5 descriptive; do not present a final selection here.",
    },
    {
        "claim_id": "IVC02",
        "manuscript_location": "IV-C. Compression Accuracy Trade-offs",
        "claim_role": "fragility",
        "claim_text": (
            "Three Phase 5 groups triggered fragility in at least one run: "
            "TinyML-MLP P50-noFT, TinyML-MLP DQ, and Compact-DNN P50-noFT."
        ),
        "evidence_assets": ["table_03_compression_results"],
        "claim_strength": "quantitative",
        "drafting_rule": "Do not generalize this fragility label beyond the locked gate.",
    },
    {
        "claim_id": "IVD01",
        "manuscript_location": "IV-D. CPU Deployment Benchmarks",
        "claim_role": "deployment",
        "claim_text": (
            "Deployment measurements use isolated single-thread CPU runs and report "
            "batch-1 latency, batch-32 throughput, state size, and peak inference RSS."
        ),
        "evidence_assets": ["table_04_deployment_benchmarks"],
        "claim_strength": "descriptive",
        "drafting_rule": "State that validation and test data were not accessed by the benchmark.",
    },
    {
        "claim_id": "IVE01",
        "manuscript_location": "IV-E. Statistical Analysis",
        "claim_role": "inference",
        "claim_text": (
            "All 16 omnibus tests reject at the nominal level, while none of the "
            "160 exact paired post-hoc comparisons reject after Holm correction."
        ),
        "evidence_assets": ["table_05_statistical_analysis"],
        "claim_strength": "quantitative_inferential",
        "drafting_rule": "Do not interpret zero post-hoc rejections as equivalence.",
    },
    {
        "claim_id": "IVE02",
        "manuscript_location": "IV-E. Statistical Analysis",
        "claim_role": "limitation",
        "claim_text": (
            "With five paired seeds, the minimum attainable two-sided exact p-value "
            "is 0.0625, limiting post-hoc inferential resolution."
        ),
        "evidence_assets": ["table_05_statistical_analysis"],
        "claim_strength": "quantitative_inferential",
        "drafting_rule": "Frame as a small-sample limitation.",
    },
    {
        "claim_id": "IVF01",
        "manuscript_location": "IV-F. Multi-objective Model-Family Selection",
        "claim_role": "selection",
        "claim_text": (
            "The balanced profile evaluates 18 compressed candidates, retains 11 "
            "eligible candidates and five Pareto candidates, and selects "
            "TinyML-MLP P50-QAT."
        ),
        "evidence_assets": ["figure_04_pareto_selection"],
        "claim_strength": "quantitative",
        "drafting_rule": "Preserve the predeclared ordering rule and profile distinction.",
    },
    {
        "claim_id": "IVG01",
        "manuscript_location": "IV-G. Input Robustness",
        "claim_role": "robustness",
        "claim_text": (
            "The robustness stage contains 27 evaluations across three models and "
            "nine input conditions; five severe-fragility evaluations all occur for HGB."
        ),
        "evidence_assets": [
            "figure_05_input_robustness",
            "table_06_robustness_results",
        ],
        "claim_strength": "quantitative",
        "drafting_rule": "Report model-specific worst conditions exactly from the locked table.",
    },
    {
        "claim_id": "V01",
        "manuscript_location": "V. Reproducibility and Release Integrity",
        "claim_role": "reproducibility",
        "claim_text": (
            "The locked deployment family is TinyML-MLP P50-QAT with static-int8 "
            "representation and canonical seed 2026."
        ),
        "evidence_assets": ["table_07_reproducibility_release"],
        "claim_strength": "quantitative",
        "drafting_rule": "Use exact model-family label and seed.",
    },
    {
        "claim_id": "V02",
        "manuscript_location": "V. Reproducibility and Release Integrity",
        "claim_role": "reproducibility",
        "claim_text": (
            "Source/bundle outputs and serialization round-trip are exact, and the "
            "release rebuild is deterministic with locked SHA-256 identifiers."
        ),
        "evidence_assets": ["table_07_reproducibility_release"],
        "claim_strength": "integrity",
        "drafting_rule": "Copy hashes exactly from the locked table; never retype from memory.",
    },
    {
        "claim_id": "VI01",
        "manuscript_location": "VI. Discussion",
        "claim_role": "interpretation",
        "claim_text": (
            "The combined evidence supports a trade-off interpretation rather than a "
            "single-metric winner: generalization, accuracy, model size, latency, "
            "throughput, inferential uncertainty, and perturbation sensitivity must "
            "be interpreted jointly."
        ),
        "evidence_assets": [
            "figure_02_lodo_generalization",
            "figure_03_compression_tradeoff",
            "table_04_deployment_benchmarks",
            "table_05_statistical_analysis",
            "figure_04_pareto_selection",
            "table_06_robustness_results",
        ],
        "claim_strength": "interpretive",
        "drafting_rule": "Do not introduce new quantitative claims in Discussion.",
    },
    {
        "claim_id": "VII01",
        "manuscript_location": "VII. Limitations and Threats to Validity",
        "claim_role": "limitation",
        "claim_text": (
            "Primary limitations include N-BaIoT-only primary evidence, five-seed "
            "inferential resolution, hardware-specific CPU benchmarking, two-class "
            "held-out devices, and deferred optional external-dataset validation."
        ),
        "evidence_assets": [
            "table_01_dataset_protocol",
            "table_04_deployment_benchmarks",
            "table_05_statistical_analysis",
        ],
        "claim_strength": "limitation",
        "drafting_rule": "State clearly that Phase 12 was not completed.",
    },
    {
        "claim_id": "VIII01",
        "manuscript_location": "VIII. Conclusion",
        "claim_role": "conclusion",
        "claim_text": (
            "The conclusion may summarize the locked device-generalization, "
            "compression/deployment, selection, robustness, and reproducibility "
            "findings without adding any new metric or experiment."
        ),
        "evidence_assets": [
            "figure_02_lodo_generalization",
            "figure_04_pareto_selection",
            "table_06_robustness_results",
            "table_07_reproducibility_release",
        ],
        "claim_strength": "summary",
        "drafting_rule": "No new numerical claim is allowed.",
    },
]

claim_checks: dict[str, bool] = {}

for claim in claims:
    evidence = claim["evidence_assets"]
    claim_checks[f"{claim['claim_id']}_has_evidence"] = len(evidence) >= 1
    claim_checks[f"{claim['claim_id']}_assets_exist"] = all(
        asset_id in asset_ids
        for asset_id in evidence
    )
    claim_checks[f"{claim['claim_id']}_has_rule"] = bool(
        claim["drafting_rule"].strip()
    )

failed_claim_checks = [
    name for name, passed in claim_checks.items() if not passed
]

if failed_claim_checks:
    raise RuntimeError(
        "Claim/evidence validation failed: "
        + ", ".join(failed_claim_checks)
    )

location_order = {
    "Abstract": 0,
    "I. Introduction": 1,
    "III. Methodology and Evaluation Protocol": 3,
    "IV-A. Strict Leave-One-Device-Out Generalization": 4,
    "IV-B. Classical Baseline Performance": 5,
    "IV-C. Compression Accuracy Trade-offs": 6,
    "IV-D. CPU Deployment Benchmarks": 7,
    "IV-E. Statistical Analysis": 8,
    "IV-F. Multi-objective Model-Family Selection": 9,
    "IV-G. Input Robustness": 10,
    "V. Reproducibility and Release Integrity": 11,
    "VI. Discussion": 12,
    "VII. Limitations and Threats to Validity": 13,
    "VIII. Conclusion": 14,
}

claims_sorted = sorted(
    claims,
    key=lambda row: (
        location_order[row["manuscript_location"]],
        row["claim_id"],
    ),
)

plan = {
    "status": "locked",
    "phase": 13,
    "artifact_name": "manuscript_claim_evidence_plan",
    "protocol_version": "phase13_manuscript_claim_evidence_v3_3",
    "locked_at_utc": utc_now(),
    "claim_count": len(claims_sorted),
    "claims": claims_sorted,
    "global_drafting_rules": [
        "Every quantitative manuscript statement must trace to a locked Phase 13 asset.",
        "Do not recompute metrics during drafting.",
        "Do not convert descriptive ranking into inferential superiority.",
        "Do not interpret non-significant Holm post-hoc results as equivalence.",
        "Do not use robustness outcomes to retroactively change model-family selection.",
        "Do not claim external-dataset validation was completed.",
        "Use exact locked hashes only in the reproducibility section.",
        "External literature claims in Introduction/Related Work require separate citations and are outside this locked internal-evidence plan.",
    ],
    "source_structure_plan": {
        "path": str(STRUCTURE_PLAN),
        "sha256": sha256_file(STRUCTURE_PLAN),
    },
    "source_asset_manifest": {
        "path": str(ASSET_MANIFEST),
        "sha256": sha256_file(ASSET_MANIFEST),
    },
    "entry_checks": entry_checks,
    "claim_checks": claim_checks,
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
            "claim_id",
            "manuscript_location",
            "claim_role",
            "claim_strength",
            "claim_text",
            "evidence_assets",
            "drafting_rule",
        ],
    )
    writer.writeheader()
    for claim in claims_sorted:
        writer.writerow(
            {
                "claim_id": claim["claim_id"],
                "manuscript_location": claim["manuscript_location"],
                "claim_role": claim["claim_role"],
                "claim_strength": claim["claim_strength"],
                "claim_text": claim["claim_text"],
                "evidence_assets": "; ".join(claim["evidence_assets"]),
                "drafting_rule": claim["drafting_rule"],
            }
        )

md_lines = [
    "# Phase 13 Manuscript Claim–Evidence Plan",
    "",
    f"Locked claims: {len(claims_sorted)}",
    "",
]

current_location = None

for claim in claims_sorted:
    if claim["manuscript_location"] != current_location:
        current_location = claim["manuscript_location"]
        md_lines.extend(
            [
                f"## {current_location}",
                "",
            ]
        )

    md_lines.extend(
        [
            f"### {claim['claim_id']} — {claim['claim_role']}",
            "",
            claim["claim_text"],
            "",
            "**Evidence:** " + ", ".join(claim["evidence_assets"]),
            "",
            f"**Drafting rule:** {claim['drafting_rule']}",
            "",
        ]
    )

md_lines.extend(
    [
        "## Global drafting rules",
        "",
    ]
)

for rule in plan["global_drafting_rules"]:
    md_lines.append(f"- {rule}")

md_lines.append("")

atomic_text(PLAN_MD, "\n".join(md_lines))

lock = {
    "status": "locked",
    "phase": 13,
    "artifact_name": "manuscript_claim_evidence_plan",
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
    "claim_count": len(claims_sorted),
    "all_claims_have_locked_evidence": True,
    "phase12_external_dataset": "deferred_optional",
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript_drafting": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 100)
print("PHASE 13 MANUSCRIPT CLAIM / EVIDENCE PLAN")
print("=" * 100)
print(f"Locked claims                    : {len(claims_sorted)}")
print(f"Locked manuscript assets         : {len(asset_ids)}")
print("All claims have locked evidence  : True")
print("Phase 12 external dataset        : deferred_optional")
print()

location_counts: dict[str, int] = {}

for claim in claims_sorted:
    location_counts[claim["manuscript_location"]] = (
        location_counts.get(claim["manuscript_location"], 0) + 1
    )

for location, count in location_counts.items():
    print(f"{location:<52} : {count} claims")

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
print("PHASE 13 CLAIM-EVIDENCE PLAN LOCKED")
