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
TABLE_DIR = ASSET_ROOT / "tables"
SOURCE_DIR = ASSET_ROOT / "source_data"
CAPTION_DIR = ASSET_ROOT / "captions"
METADATA_DIR = ASSET_ROOT / "metadata"

BINDING_PATH = CONFIG / "phase13_exact_source_binding_v3_3.json"
BINDING_LOCK_PATH = AUDIT / "phase13_exact_source_binding_locked_v3_3.json"
MAPPING_PATH = CONFIG / "phase13_manuscript_column_mapping_v3_3.json"
MAPPING_LOCK_PATH = AUDIT / "phase13_manuscript_column_mapping_locked_v3_3.json"

TABLE_04_LOCK = AUDIT / "phase13_table_04_deployment_benchmarks_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_05_statistical_analysis_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_05_statistical_analysis_v3_3.md"
SOURCE_OMNIBUS_CSV = SOURCE_DIR / "table_05_statistical_omnibus_source_v3_3.csv"
SOURCE_POSTHOC_CSV = SOURCE_DIR / "table_05_statistical_posthoc_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "table_05_statistical_analysis_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_05_statistical_analysis_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_05_statistical_analysis_locked_v3_3.json"

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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def get_bound_source(binding: dict[str, Any], source_id: str) -> Path:
    registry = binding["source_registry"]

    if source_id not in registry:
        raise KeyError(f"Source ID not found in binding: {source_id}")

    record = registry[source_id]
    path = Path(record["path"])

    if not path.exists():
        raise FileNotFoundError(path)

    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Bound source changed after lock: {path}")

    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
    TABLE_04_LOCK,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for output_path in (
    TABLE_CSV,
    TABLE_MD,
    SOURCE_OMNIBUS_CSV,
    SOURCE_POSTHOC_CSV,
    CAPTION_TXT,
    METADATA_JSON,
    LOCK_JSON,
):
    if output_path.exists():
        raise FileExistsError(
            "Table 05 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_04_lock = read_json(TABLE_04_LOCK)

entry_checks = {
    "binding_locked": (
        binding.get("status") == "locked"
        and binding.get("ready_for_asset_generation") is True
        and binding.get("all_checks_passed") is True
    ),
    "binding_hash_matches": (
        binding_lock["binding"]["sha256"] == sha256_file(BINDING_PATH)
    ),
    "mapping_validated": (
        mapping.get("status") == "validated"
        and mapping.get("ready_for_asset_generation") is True
        and mapping.get("all_checks_passed") is True
    ),
    "mapping_hash_matches": (
        mapping_lock["mapping"]["sha256"] == sha256_file(MAPPING_PATH)
    ),
    "table_04_locked": (
        table_04_lock.get("status") == "locked"
        and table_04_lock.get("ready_for_manuscript") is True
        and table_04_lock.get("all_checks_passed") is True
    ),
    "table_05_mapped": any(
        asset["asset_id"] == "table_05_statistical_analysis"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 05 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

omnibus_path = get_bound_source(binding, "phase7_omnibus")
posthoc_path = get_bound_source(binding, "phase7_posthoc_verified")
summary_path = get_bound_source(binding, "phase7_summary")

omnibus_rows = read_csv_rows(omnibus_path)
posthoc_rows = read_csv_rows(posthoc_path)
summary = read_json(summary_path)

required_omnibus = {
    "endpoint_class",
    "architecture",
    "metric",
    "block_count",
    "variant_count",
    "friedman_statistic",
    "kendalls_W",
    "monte_carlo_p_plus_one",
    "monte_carlo_reject_nominal",
}

required_posthoc = {
    "endpoint_class",
    "architecture",
    "metric",
    "reference_variant",
    "candidate_variant",
    "mean_improvement_delta",
    "paired_rank_biserial_improvement",
    "exact_two_sided_p_raw",
    "Holm_adjusted_p",
    "Holm_reject_alpha_0_05",
    "all_checks_passed",
}

omnibus_headers = set(omnibus_rows[0].keys()) if omnibus_rows else set()
posthoc_headers = set(posthoc_rows[0].keys()) if posthoc_rows else set()

data_checks = {
    "omnibus_rows_16": len(omnibus_rows) == 16,
    "posthoc_rows_160": len(posthoc_rows) == 160,
    "required_omnibus_columns_present": required_omnibus.issubset(
        omnibus_headers
    ),
    "required_posthoc_columns_present": required_posthoc.issubset(
        posthoc_headers
    ),
    "summary_omnibus_16": summary["omnibus_analysis_count"] == 16,
    "summary_posthoc_160": summary["posthoc_comparison_count"] == 160,
    "summary_confirmatory_holm_zero": (
        summary["confirmatory_posthoc_Holm_rejections"] == 0
    ),
    "summary_exploratory_holm_zero": (
        summary["exploratory_posthoc_Holm_rejections"] == 0
    ),
    "minimum_raw_posthoc_p_00625": (
        abs(summary["minimum_raw_posthoc_p"] - 0.0625) <= 1e-12
    ),
    "minimum_holm_p_0625": (
        abs(summary["minimum_Holm_adjusted_p"] - 0.625) <= 1e-12
    ),
    "all_posthoc_verified": all(
        parse_bool(row["all_checks_passed"]) is True
        for row in posthoc_rows
    ),
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 05 source validation failed: "
        + ", ".join(failed_data_checks)
    )

posthoc_groups: dict[
    tuple[str, str, str],
    list[dict[str, str]],
] = {}

for row in posthoc_rows:
    key = (
        row["endpoint_class"],
        row["architecture"],
        row["metric"],
    )
    posthoc_groups.setdefault(key, []).append(row)

table_rows: list[dict[str, str]] = []

group_checks: dict[str, bool] = {}

for omnibus in omnibus_rows:
    key = (
        omnibus["endpoint_class"],
        omnibus["architecture"],
        omnibus["metric"],
    )

    group = posthoc_groups.get(key, [])

    if not group:
        raise RuntimeError(
            "No verified post-hoc rows found for omnibus stratum: "
            f"{key}"
        )

    raw_p_values = [
        float(row["exact_two_sided_p_raw"])
        for row in group
    ]

    holm_p_values = [
        float(row["Holm_adjusted_p"])
        for row in group
    ]

    holm_rejections = sum(
        parse_bool(row["Holm_reject_alpha_0_05"])
        for row in group
    )

    check_key = "::".join(key)

    group_checks[f"{check_key}_posthoc_count_10"] = (
        len(group) == 10
    )

    group_checks[f"{check_key}_reference_B0"] = all(
        row["reference_variant"] == "B0"
        for row in group
    )

    group_checks[f"{check_key}_holm_zero"] = (
        holm_rejections == 0
    )

    table_rows.append(
        {
            "Endpoint": omnibus["endpoint_class"],
            "Architecture": omnibus["architecture"],
            "Metric": omnibus["metric"],
            "Blocks": omnibus["block_count"],
            "Variants": omnibus["variant_count"],
            "Friedman statistic": (
                f"{float(omnibus['friedman_statistic']):.6f}"
            ),
            "Kendall W": (
                f"{float(omnibus['kendalls_W']):.6f}"
            ),
            "Monte Carlo p": (
                f"{float(omnibus['monte_carlo_p_plus_one']):.6f}"
            ),
            "Omnibus reject (nominal)": (
                "Yes"
                if parse_bool(omnibus["monte_carlo_reject_nominal"])
                else "No"
            ),
            "Post-hoc comparisons vs B0": str(len(group)),
            "Min exact post-hoc p": f"{min(raw_p_values):.4f}",
            "Min Holm-adjusted p": f"{min(holm_p_values):.4f}",
            "Holm rejections": str(holm_rejections),
        }
    )

failed_group_checks = [
    name for name, passed in group_checks.items() if not passed
]

if failed_group_checks:
    raise RuntimeError(
        "Table 05 post-hoc grouping validation failed: "
        + ", ".join(failed_group_checks)
    )

if len(table_rows) != 16:
    raise RuntimeError(
        f"Expected 16 manuscript rows, found {len(table_rows)}"
    )

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

table_fieldnames = [
    "Endpoint",
    "Architecture",
    "Metric",
    "Blocks",
    "Variants",
    "Friedman statistic",
    "Kendall W",
    "Monte Carlo p",
    "Omnibus reject (nominal)",
    "Post-hoc comparisons vs B0",
    "Min exact post-hoc p",
    "Min Holm-adjusted p",
    "Holm rejections",
]

with TABLE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=table_fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

with SOURCE_OMNIBUS_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(omnibus_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(omnibus_rows)

with SOURCE_POSTHOC_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(posthoc_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(posthoc_rows)

md_lines = [
    "| Endpoint | Architecture | Metric | Blocks | Variants | Friedman statistic | Kendall W | Monte Carlo p | Omnibus reject (nominal) | Post-hoc comparisons vs B0 | Min exact post-hoc p | Min Holm-adjusted p | Holm rejections |",
    "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
]

for row in table_rows:
    md_lines.append(
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|")
            for field in table_fieldnames
        )
        + " |"
    )

atomic_text(TABLE_MD, "\n".join(md_lines) + "\n")

nominal_rejections = sum(
    parse_bool(row["monte_carlo_reject_nominal"])
    for row in omnibus_rows
)

holm_rejections_total = sum(
    parse_bool(row["Holm_reject_alpha_0_05"])
    for row in posthoc_rows
)

caption = (
    "Table 5. Locked omnibus and verified post-hoc statistical analysis across "
    "the compression/deployment endpoints. Each omnibus stratum contains five "
    "paired seed blocks and eleven variants. Monte Carlo Friedman tests are "
    "reported with Kendall's W effect size. Ten exact paired post-hoc comparisons "
    "against B0 were evaluated per stratum and adjusted with Holm correction. "
    "Although all 16 omnibus tests rejected at the nominal level, none of the "
    "160 post-hoc comparisons survived Holm correction. With five paired seeds, "
    "the minimum attainable two-sided exact p-value was 0.0625; failure to reject "
    "is therefore not interpreted as equivalence."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_05_statistical_analysis",
    "generated_at_utc": utc_now(),
    "title": "Statistical Analysis",
    "table_row_count": len(table_rows),
    "omnibus_analysis_count": len(omnibus_rows),
    "posthoc_comparison_count": len(posthoc_rows),
    "nominal_omnibus_rejection_count": nominal_rejections,
    "holm_posthoc_rejection_count": holm_rejections_total,
    "minimum_raw_posthoc_p": summary["minimum_raw_posthoc_p"],
    "minimum_holm_adjusted_p": summary["minimum_Holm_adjusted_p"],
    "small_sample_interpretation": summary["small_sample_interpretation"],
    "source_files": [
        {
            "source_id": "phase7_omnibus",
            "path": str(omnibus_path),
            "sha256": sha256_file(omnibus_path),
        },
        {
            "source_id": "phase7_posthoc_verified",
            "path": str(posthoc_path),
            "sha256": sha256_file(posthoc_path),
        },
        {
            "source_id": "phase7_summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
    ],
    "outputs": {
        "csv": {
            "path": str(TABLE_CSV),
            "sha256": sha256_file(TABLE_CSV),
            "size_bytes": TABLE_CSV.stat().st_size,
        },
        "markdown": {
            "path": str(TABLE_MD),
            "sha256": sha256_file(TABLE_MD),
            "size_bytes": TABLE_MD.stat().st_size,
        },
        "source_omnibus_csv": {
            "path": str(SOURCE_OMNIBUS_CSV),
            "sha256": sha256_file(SOURCE_OMNIBUS_CSV),
            "size_bytes": SOURCE_OMNIBUS_CSV.stat().st_size,
        },
        "source_posthoc_csv": {
            "path": str(SOURCE_POSTHOC_CSV),
            "sha256": sha256_file(SOURCE_POSTHOC_CSV),
            "size_bytes": SOURCE_POSTHOC_CSV.stat().st_size,
        },
        "caption": {
            "path": str(CAPTION_TXT),
            "sha256": sha256_file(CAPTION_TXT),
            "size_bytes": CAPTION_TXT.stat().st_size,
        },
    },
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "group_checks": group_checks,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "table_05_statistical_analysis",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "table_row_count": 16,
    "omnibus_analysis_count": 16,
    "posthoc_comparison_count": 160,
    "nominal_omnibus_rejection_count": nominal_rejections,
    "holm_posthoc_rejection_count": holm_rejections_total,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 05 - STATISTICAL ANALYSIS")
print("=" * 96)
print(f"Table rows                       : {len(table_rows)}")
print(f"Omnibus analyses                 : {len(omnibus_rows)}")
print(f"Post-hoc comparisons             : {len(posthoc_rows)}")
print(f"Nominal omnibus rejections       : {nominal_rejections}")
print(f"Holm post-hoc rejections         : {holm_rejections_total}")
print(
    "Minimum exact post-hoc p        : "
    f"{summary['minimum_raw_posthoc_p']:.4f}"
)
print(
    "Minimum Holm-adjusted p         : "
    f"{summary['minimum_Holm_adjusted_p']:.4f}"
)
print(
    "Minimum possible exact p (n=5) : "
    f"{summary['small_sample_interpretation']['minimum_possible_two_sided_exact_p']:.4f}"
)
print()
print(f"CSV                              : {TABLE_CSV}")
print(f"Markdown                         : {TABLE_MD}")
print(f"Omnibus source CSV               : {SOURCE_OMNIBUS_CSV}")
print(f"Post-hoc source CSV              : {SOURCE_POSTHOC_CSV}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model inference performed        : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 TABLE 05 GENERATED AND LOCKED")
