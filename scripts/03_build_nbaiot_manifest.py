"""
N-BaIoT sınıf ve cihaz manifesti oluşturma betiği.

Bu betik, daha önce oluşturulan dosya envanterini kullanarak:

- Çok sınıflı etiketleri oluşturur.
- Her sınıfın örnek sayısını hesaplar.
- Her cihazdaki sınıf dağılımını çıkarır.
- Eksik cihaz-sınıf kombinasyonlarını belirler.
- Grup tabanlı veri bölme tasarımı için raporlar üretir.

Ham CSV dosyalarını tekrar okumaz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_file_inventory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


def normalize_text(value: object) -> str:
    """Etiket metinlerini standart biçime dönüştürür."""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def create_class_label(
    attack_family: object,
    attack_type: object,
) -> str:
    """
    Saldırı ailesi ve saldırı türünden nihai sınıf etiketini üretir.

    Örnekler:
        benign
        gafgyt_combo
        gafgyt_junk
        mirai_ack
        mirai_udpplain
    """

    family = normalize_text(attack_family)
    attack = normalize_text(attack_type)

    if family == "benign" or attack == "benign":
        return "benign"

    return f"{family}_{attack}"


def validate_inventory(inventory: pd.DataFrame) -> None:
    """Envanter dosyasının gerekli alanlarını doğrular."""

    required_columns = {
        "device",
        "attack_family",
        "attack_type",
        "relative_path",
        "file_name",
        "file_size_bytes",
        "row_count",
        "column_count",
        "status",
    }

    missing_columns = (
        required_columns
        - set(inventory.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Envanter dosyasında gerekli sütunlar eksik: "
            f"{missing_text}"
        )

    failed_files = inventory[
        inventory["status"] != "success"
    ]

    if not failed_files.empty:
        raise ValueError(
            f"Envanterde {len(failed_files)} hatalı dosya bulunuyor."
        )


def dataframe_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """DataFrame'i JSON uyumlu kayıt listesine dönüştürür."""

    return json.loads(
        dataframe.to_json(
            orient="records",
            force_ascii=False,
        )
    )


def main() -> None:
    """Ana program akışı."""

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            "Dosya envanteri bulunamadı: "
            f"{INVENTORY_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory = pd.read_csv(
        INVENTORY_FILE
    )

    validate_inventory(inventory)

    inventory["device"] = (
        inventory["device"]
        .astype(str)
        .str.strip()
    )

    inventory["attack_family"] = (
        inventory["attack_family"]
        .map(normalize_text)
    )

    inventory["attack_type"] = (
        inventory["attack_type"]
        .map(normalize_text)
    )

    inventory["row_count"] = (
        pd.to_numeric(
            inventory["row_count"],
            errors="raise",
        )
        .astype("int64")
    )

    inventory["file_size_bytes"] = (
        pd.to_numeric(
            inventory["file_size_bytes"],
            errors="raise",
        )
        .astype("int64")
    )

    inventory["class_label"] = [
        create_class_label(
            attack_family=family,
            attack_type=attack_type,
        )
        for family, attack_type in zip(
            inventory["attack_family"],
            inventory["attack_type"],
            strict=True,
        )
    ]

    # ------------------------------------------------------
    # Sınıf dağılımı
    # ------------------------------------------------------

    class_distribution = (
        inventory.groupby(
            [
                "class_label",
                "attack_family",
                "attack_type",
            ],
            as_index=False,
        )
        .agg(
            file_count=(
                "relative_path",
                "count",
            ),
            sample_count=(
                "row_count",
                "sum",
            ),
            size_bytes=(
                "file_size_bytes",
                "sum",
            ),
            device_count=(
                "device",
                "nunique",
            ),
        )
    )

    total_samples = int(
        class_distribution[
            "sample_count"
        ].sum()
    )

    class_distribution[
        "sample_percentage"
    ] = (
        class_distribution["sample_count"]
        / total_samples
        * 100
    )

    class_distribution[
        "size_gb"
    ] = (
        class_distribution["size_bytes"]
        / (1024**3)
    )

    class_distribution = (
        class_distribution.sort_values(
            "sample_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Cihaz özeti
    # ------------------------------------------------------

    device_distribution = (
        inventory.groupby(
            "device",
            as_index=False,
        )
        .agg(
            file_count=(
                "relative_path",
                "count",
            ),
            sample_count=(
                "row_count",
                "sum",
            ),
            class_count=(
                "class_label",
                "nunique",
            ),
            size_bytes=(
                "file_size_bytes",
                "sum",
            ),
        )
    )

    device_distribution[
        "sample_percentage"
    ] = (
        device_distribution["sample_count"]
        / total_samples
        * 100
    )

    device_distribution[
        "size_gb"
    ] = (
        device_distribution["size_bytes"]
        / (1024**3)
    )

    device_distribution = (
        device_distribution.sort_values(
            "sample_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Cihaz × sınıf örnek matrisi
    # ------------------------------------------------------

    device_class_matrix = (
        inventory.pivot_table(
            index="device",
            columns="class_label",
            values="row_count",
            aggfunc="sum",
            fill_value=0,
        )
        .astype("int64")
        .sort_index()
    )

    preferred_class_order = [
        "benign",
        "gafgyt_combo",
        "gafgyt_junk",
        "gafgyt_scan",
        "gafgyt_tcp",
        "gafgyt_udp",
        "mirai_ack",
        "mirai_scan",
        "mirai_syn",
        "mirai_udp",
        "mirai_udpplain",
    ]

    existing_preferred = [
        label
        for label in preferred_class_order
        if label in device_class_matrix.columns
    ]

    remaining_classes = sorted(
        set(device_class_matrix.columns)
        - set(existing_preferred)
    )

    device_class_matrix = (
        device_class_matrix[
            existing_preferred
            + remaining_classes
        ]
    )

    class_coverage_matrix = (
        device_class_matrix > 0
    ).astype("int8")

    # ------------------------------------------------------
    # Eksik sınıfları belirle
    # ------------------------------------------------------

    all_classes = list(
        device_class_matrix.columns
    )

    missing_classes_by_device: dict[
        str,
        list[str],
    ] = {}

    for device_name, row in (
        class_coverage_matrix.iterrows()
    ):
        missing_classes_by_device[
            str(device_name)
        ] = [
            class_name
            for class_name in all_classes
            if int(row[class_name]) == 0
        ]

    device_count_by_class = {
        class_name: int(
            class_coverage_matrix[
                class_name
            ].sum()
        )
        for class_name in all_classes
    }

    # ------------------------------------------------------
    # Dosyaları kaydet
    # ------------------------------------------------------

    manifest_file = (
        OUTPUT_DIR
        / "nbaiot_manifest.csv"
    )

    class_distribution_file = (
        OUTPUT_DIR
        / "nbaiot_class_distribution.csv"
    )

    device_distribution_file = (
        OUTPUT_DIR
        / "nbaiot_device_distribution.csv"
    )

    device_class_file = (
        OUTPUT_DIR
        / "nbaiot_device_class_matrix.csv"
    )

    class_coverage_file = (
        OUTPUT_DIR
        / "nbaiot_device_class_coverage.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "nbaiot_manifest_summary.json"
    )

    inventory.to_csv(
        manifest_file,
        index=False,
        encoding="utf-8",
    )

    class_distribution.to_csv(
        class_distribution_file,
        index=False,
        encoding="utf-8",
    )

    device_distribution.to_csv(
        device_distribution_file,
        index=False,
        encoding="utf-8",
    )

    device_class_matrix.to_csv(
        device_class_file,
        encoding="utf-8",
    )

    class_coverage_matrix.to_csv(
        class_coverage_file,
        encoding="utf-8",
    )

    summary = {
        "total_samples": total_samples,
        "total_files": int(len(inventory)),
        "device_count": int(
            inventory["device"].nunique()
        ),
        "class_count": int(
            inventory["class_label"].nunique()
        ),
        "attack_family_count": int(
            inventory[
                "attack_family"
            ].nunique()
        ),
        "classes": all_classes,
        "devices": sorted(
            inventory["device"]
            .unique()
            .tolist()
        ),
        "class_distribution": dataframe_records(
            class_distribution
        ),
        "device_distribution": dataframe_records(
            device_distribution
        ),
        "device_count_by_class": (
            device_count_by_class
        ),
        "missing_classes_by_device": (
            missing_classes_by_device
        ),
    }

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

    # ------------------------------------------------------
    # Terminal özeti
    # ------------------------------------------------------

    print("=" * 78)
    print("N-BaIoT Sınıf ve Cihaz Manifesti")
    print("=" * 78)
    print(f"Dosya sayısı   : {len(inventory)}")
    print(f"Toplam örnek   : {total_samples:,}")
    print(
        "Cihaz sayısı   : "
        f"{inventory['device'].nunique()}"
    )
    print(
        "Sınıf sayısı   : "
        f"{inventory['class_label'].nunique()}"
    )
    print()

    display_columns = [
        "class_label",
        "sample_count",
        "sample_percentage",
        "file_count",
        "device_count",
    ]

    print(
        class_distribution[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "sample_percentage": (
                    lambda value: f"{value:.4f}"
                )
            },
        )
    )

    print()
    print("Cihaz başına eksik sınıflar:")

    for device_name in sorted(
        missing_classes_by_device
    ):
        missing = (
            missing_classes_by_device[
                device_name
            ]
        )

        missing_text = (
            ", ".join(missing)
            if missing
            else "Yok"
        )

        print(
            f"- {device_name}: "
            f"{missing_text}"
        )

    print()
    print(f"Manifest      : {manifest_file}")
    print(
        "Sınıf özeti   : "
        f"{class_distribution_file}"
    )
    print(
        "Cihaz özeti   : "
        f"{device_distribution_file}"
    )
    print(
        "Cihaz-sınıf   : "
        f"{device_class_file}"
    )
    print(
        "Kapsama       : "
        f"{class_coverage_file}"
    )
    print(f"JSON özet     : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()