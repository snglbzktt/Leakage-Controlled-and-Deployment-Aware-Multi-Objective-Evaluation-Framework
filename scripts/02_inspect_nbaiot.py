"""
N-BaIoT ham veri envanteri.

Bu betik:
- CSV dosyalarını bulur.
- Dosya yollarından cihaz ve saldırı etiketlerini üretir.
- Dosyaları tamamen RAM'e yüklemeden satır sayılarını hesaplar.
- Sütun şemalarını karşılaştırır.
- CSV ve JSON biçiminde veri envanteri oluşturur.

Not:
Bu aşamada eksik değer, sonsuz değer ve satır tekrarları tam olarak
taranmaz. Bu kontroller sonraki veri doğrulama aşamasında yapılacaktır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nbaiot_unpacked"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


def count_file_lines(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    """
    Bir dosyadaki satır sayısını belleğe tamamen yüklemeden hesaplar.

    N-BaIoT CSV dosyalarında her kayıt tek fiziksel satırda bulunduğu
    için bu yöntem hızlı bir satır sayımı sağlar.
    """

    line_count = 0
    last_byte = b""

    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)

            if not chunk:
                break

            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if file_path.stat().st_size > 0 and last_byte != b"\n":
        line_count += 1

    return line_count


def calculate_sha256(file_path: Path) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(8 * 1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def create_schema_fingerprint(columns: list[str]) -> str:
    """Sütun sırası ve adları için kısa bir şema kimliği üretir."""

    schema_text = "\x1f".join(columns)

    return hashlib.sha256(
        schema_text.encode("utf-8")
    ).hexdigest()


def infer_labels(
    file_path: Path,
    data_root: Path,
) -> dict[str, str]:
    """
    Dosya yolundan cihaz, saldırı ailesi ve saldırı türünü çıkarır.
    """

    relative_path = file_path.relative_to(data_root)
    relative_parts = list(relative_path.parts)

    device_name = (
        relative_parts[0]
        if relative_parts
        else "unknown"
    )

    lowercase_parts = [
        part.lower()
        for part in relative_parts
    ]

    file_stem = file_path.stem.lower()

    if "benign" in file_stem:
        attack_family = "benign"
        attack_type = "benign"

    elif any(
        "gafgyt" in part
        for part in lowercase_parts
    ):
        attack_family = "gafgyt"
        attack_type = file_stem

    elif any(
        "mirai" in part
        for part in lowercase_parts
    ):
        attack_family = "mirai"
        attack_type = file_stem

    else:
        attack_family = "unknown"
        attack_type = file_stem

    return {
        "device": device_name,
        "attack_family": attack_family,
        "attack_type": attack_type,
        "relative_path": relative_path.as_posix(),
    }


def inspect_csv_file(
    file_path: Path,
    data_root: Path,
    hash_files: bool,
) -> dict[str, Any]:
    """
    Tek bir CSV dosyasının metadata bilgilerini çıkarır.
    """

    labels = infer_labels(
        file_path=file_path,
        data_root=data_root,
    )

    record: dict[str, Any] = {
        **labels,
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "file_size_mb": round(
            file_path.stat().st_size / (1024**2),
            4,
        ),
        "row_count": None,
        "column_count": None,
        "schema_fingerprint": None,
        "duplicate_column_count": None,
        "sha256": None,
        "status": "success",
        "error": None,
    }

    try:
        header_frame = pd.read_csv(
            file_path,
            nrows=0,
        )

        columns = [
            str(column).strip()
            for column in header_frame.columns
        ]

        physical_line_count = count_file_lines(
            file_path
        )

        # Birinci satır sütun başlıklarını içerir.
        data_row_count = max(
            physical_line_count - 1,
            0,
        )

        duplicate_column_count = (
            len(columns)
            - len(set(columns))
        )

        record["row_count"] = data_row_count
        record["column_count"] = len(columns)
        record["duplicate_column_count"] = (
            duplicate_column_count
        )
        record["schema_fingerprint"] = (
            create_schema_fingerprint(columns)
        )

        if hash_files:
            record["sha256"] = calculate_sha256(
                file_path
            )

    except Exception as error:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return record


def build_summary(
    inventory: pd.DataFrame,
    data_root: Path,
) -> dict[str, Any]:
    """Dosya envanterinden genel veri seti özetini oluşturur."""

    successful = inventory[
        inventory["status"] == "success"
    ].copy()

    errors = inventory[
        inventory["status"] == "error"
    ].copy()

    label_summary = (
        successful.groupby(
            [
                "attack_family",
                "attack_type",
            ],
            dropna=False,
        )
        .agg(
            file_count=("file_name", "count"),
            row_count=("row_count", "sum"),
            size_bytes=("file_size_bytes", "sum"),
        )
        .reset_index()
    )

    device_summary = (
        successful.groupby(
            "device",
            dropna=False,
        )
        .agg(
            file_count=("file_name", "count"),
            row_count=("row_count", "sum"),
            size_bytes=("file_size_bytes", "sum"),
        )
        .reset_index()
    )

    schema_count = int(
        successful[
            "schema_fingerprint"
        ].nunique()
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_root": str(data_root.resolve()),
        "total_csv_files": int(len(inventory)),
        "successful_files": int(len(successful)),
        "error_files": int(len(errors)),
        "total_rows": int(
            successful["row_count"].fillna(0).sum()
        ),
        "total_size_bytes": int(
            successful["file_size_bytes"].sum()
        ),
        "total_size_gb": round(
            successful["file_size_bytes"].sum()
            / (1024**3),
            4,
        ),
        "device_count": int(
            successful["device"].nunique()
        ),
        "devices": sorted(
            successful["device"]
            .dropna()
            .unique()
            .tolist()
        ),
        "attack_family_count": int(
            successful[
                "attack_family"
            ].nunique()
        ),
        "attack_families": sorted(
            successful[
                "attack_family"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
        "attack_type_count": int(
            successful[
                "attack_type"
            ].nunique()
        ),
        "column_count_min": (
            int(successful["column_count"].min())
            if not successful.empty
            else None
        ),
        "column_count_max": (
            int(successful["column_count"].max())
            if not successful.empty
            else None
        ),
        "schema_fingerprint_count": schema_count,
        "schema_consistent": schema_count == 1,
        "duplicate_column_files": int(
            (
                successful[
                    "duplicate_column_count"
                ].fillna(0)
                > 0
            ).sum()
        ),
        "label_summary": (
            label_summary.to_dict(
                orient="records"
            )
        ),
        "device_summary": (
            device_summary.to_dict(
                orient="records"
            )
        ),
        "errors": (
            errors[
                [
                    "relative_path",
                    "error",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "row_count_method": (
            "Physical line count minus one CSV header row"
        ),
    }

    return summary


def save_reference_schema(
    inventory: pd.DataFrame,
    data_root: Path,
    output_file: Path,
) -> None:
    """
    İlk başarılı CSV dosyasının sütun şemasını kaydeder.
    """

    successful = inventory[
        inventory["status"] == "success"
    ]

    if successful.empty:
        return

    reference_relative_path = successful.iloc[
        0
    ]["relative_path"]

    reference_file = (
        data_root
        / Path(reference_relative_path)
    )

    reference_columns = pd.read_csv(
        reference_file,
        nrows=0,
    ).columns

    schema_frame = pd.DataFrame(
        {
            "column_index": range(
                len(reference_columns)
            ),
            "column_name": [
                str(column).strip()
                for column in reference_columns
            ],
        }
    )

    schema_frame.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Komut satırı argümanlarını oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT CSV dosyaları için veri "
            "envanteri oluşturur."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Çıkarılmış N-BaIoT veri klasörü.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Raporların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--hash-files",
        action="store_true",
        help=(
            "Her CSV dosyası için SHA-256 hesaplar. "
            "Bu seçenek işlemi uzatır."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_root}"
        )

    csv_files = sorted(
        data_root.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı: {data_root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("N-BaIoT Veri Envanteri")
    print("=" * 70)
    print(f"Veri klasörü : {data_root}")
    print(f"CSV sayısı   : {len(csv_files)}")
    print(f"SHA-256      : {args.hash_files}")
    print("=" * 70)

    records: list[dict[str, Any]] = []

    for csv_file in tqdm(
        csv_files,
        desc="CSV dosyaları inceleniyor",
        unit="dosya",
    ):
        records.append(
            inspect_csv_file(
                file_path=csv_file,
                data_root=data_root,
                hash_files=args.hash_files,
            )
        )

    inventory = pd.DataFrame(records)

    inventory = inventory.sort_values(
        by=[
            "device",
            "attack_family",
            "attack_type",
            "relative_path",
        ]
    ).reset_index(drop=True)

    inventory_file = (
        output_dir
        / "nbaiot_file_inventory.csv"
    )

    summary_file = (
        output_dir
        / "nbaiot_dataset_summary.json"
    )

    schema_file = (
        output_dir
        / "nbaiot_reference_schema.csv"
    )

    inventory.to_csv(
        inventory_file,
        index=False,
        encoding="utf-8",
    )

    summary = build_summary(
        inventory=inventory,
        data_root=data_root,
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            summary,
            file_handle,
            indent=2,
            ensure_ascii=False,
        )

    save_reference_schema(
        inventory=inventory,
        data_root=data_root,
        output_file=schema_file,
    )

    print()
    print("=" * 70)
    print("Envanter tamamlandı")
    print("=" * 70)
    print(
        f"Başarılı dosya : "
        f"{summary['successful_files']}"
    )
    print(
        f"Hatalı dosya   : "
        f"{summary['error_files']}"
    )
    print(
        f"Toplam örnek   : "
        f"{summary['total_rows']:,}"
    )
    print(
        f"Toplam boyut   : "
        f"{summary['total_size_gb']} GB"
    )
    print(
        f"Cihaz sayısı   : "
        f"{summary['device_count']}"
    )
    print(
        f"Sütun aralığı  : "
        f"{summary['column_count_min']}–"
        f"{summary['column_count_max']}"
    )
    print(
        f"Şema tutarlı   : "
        f"{summary['schema_consistent']}"
    )
    print()
    print(f"Envanter : {inventory_file}")
    print(f"Özet     : {summary_file}")
    print(f"Şema     : {schema_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()