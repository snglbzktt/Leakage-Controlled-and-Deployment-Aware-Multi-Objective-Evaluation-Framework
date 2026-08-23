"""
34. Aşama — IEEE Access makale iskeleti ve kanıt haritası.

Bu betik 32. ve 33. aşama doğrulanmış raporlarını kullanır.
Yeni eğitim, test değerlendirmesi veya istatistiksel test yapmaz.
Uydurma literatür atfı üretmez.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results" / "reports"
STAGE33 = ROOT / "results" / "publication" / "stage33"
OUTPUT = ROOT / "results" / "publication" / "stage34"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-directory", type=Path, default=REPORTS)
    parser.add_argument("--stage33-directory", type=Path, default=STAGE33)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def load_inputs(report_dir: Path, stage33_dir: Path) -> dict[str, Any]:
    paths = {
        "summary": require(
            report_dir / "nbaiot_family3_overall_statistical_summary.json"
        ),
        "performance": require(stage33_dir / "table_1_performance_summary.csv"),
        "efficiency": require(stage33_dir / "table_2_efficiency_summary.csv"),
        "statistics": require(stage33_dir / "table_3_statistical_summary.csv"),
        "fnr": require(stage33_dir / "table_4_classwise_fnr_summary.csv"),
        "pareto": require(stage33_dir / "table_5_pareto_optimal_solutions.csv"),
        "manifest33": require(stage33_dir / "stage33_manifest.json"),
    }

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    manifest33 = json.loads(paths["manifest33"].read_text(encoding="utf-8"))

    if not summary.get("validation_passed", False):
        raise RuntimeError("32. aşama doğrulaması geçmemiş.")
    if not manifest33.get("validation_passed", False):
        raise RuntimeError("33. aşama doğrulaması geçmemiş.")
    if manifest33.get("mcu_measurement", True):
        raise RuntimeError("33. aşama manifesti MCU ölçümü içeriyor.")
    if manifest33.get("test_re_evaluated", True):
        raise RuntimeError("33. aşamada test yeniden değerlendirilmiş.")

    frames = {
        "performance": pd.read_csv(paths["performance"]),
        "efficiency": pd.read_csv(paths["efficiency"]),
        "statistics": pd.read_csv(paths["statistics"]),
        "fnr": pd.read_csv(paths["fnr"]),
        "pareto": pd.read_csv(paths["pareto"]),
    }

    expected_counts = {
        "performance": 30,
        "efficiency": 30,
        "statistics": 3,
        "fnr": 30,
    }
    for name, expected in expected_counts.items():
        if len(frames[name]) != expected:
            raise RuntimeError(
                f"{name} satır sayısı {expected} değil: {len(frames[name])}"
            )

    return {
        "paths": paths,
        "summary": summary,
        "manifest33": manifest33,
        **frames,
    }


def best_row(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return frame.sort_values(
        [column, "Model", "Varyant"],
        ascending=[ascending, True, True],
    ).iloc[0]


def headline(inputs: dict[str, Any]) -> dict[str, Any]:
    efficiency = inputs["efficiency"].copy()
    numeric = [
        "Macro F1",
        "Artifact Boyutu (KiB)",
        "Medyan Gecikme (ms)",
        "Throughput (örnek/s)",
    ]
    for column in numeric:
        efficiency[column] = pd.to_numeric(efficiency[column], errors="raise")

    accuracy = best_row(efficiency, "Macro F1", False)
    size = best_row(efficiency, "Artifact Boyutu (KiB)", True)
    latency = best_row(efficiency, "Medyan Gecikme (ms)", True)
    throughput = best_row(efficiency, "Throughput (örnek/s)", False)

    stats = {}
    for _, row in inputs["statistics"].iterrows():
        stats[str(row["Model"])] = {
            "chi_square": float(row["Friedman χ²"]),
            "df": int(row["Serbestlik Derecesi"]),
            "p": float(row["Friedman p"]),
            "w": float(row["Kendall W"]),
            "magnitude": str(row["Etki Büyüklüğü"]),
            "holm_significant": int(
                row["Holm Sonrası Anlamlı İkili Karşılaştırma"]
            ),
        }

    global_pareto = inputs["pareto"][
        inputs["pareto"]["Scope"].astype(str) == "Global"
    ]

    return {
        "best_accuracy": {
            "model": str(accuracy["Model"]),
            "variant": str(accuracy["Varyant"]),
            "value": float(accuracy["Macro F1"]),
        },
        "smallest_artifact": {
            "model": str(size["Model"]),
            "variant": str(size["Varyant"]),
            "value": float(size["Artifact Boyutu (KiB)"]),
        },
        "lowest_latency": {
            "model": str(latency["Model"]),
            "variant": str(latency["Varyant"]),
            "value": float(latency["Medyan Gecikme (ms)"]),
        },
        "highest_throughput": {
            "model": str(throughput["Model"]),
            "variant": str(throughput["Varyant"]),
            "value": float(throughput["Throughput (örnek/s)"]),
        },
        "friedman": stats,
        "global_pareto_count": int(len(global_pareto)),
    }


def section_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["1", "Giriş", "Problemi, boşluğu ve katkıları tanımlamak",
             "Stage32 summary", "Yok", "Literatür atıfları gerekli"],
            ["2", "İlgili Çalışmalar", "IoT IDS, TinyML, budama ve kuantizasyon literatürü",
             "Hakemli kaynaklar", "Karşılaştırma tablosu", "35. aşamada yazılacak"],
            ["3", "Veri Seti ve Ön İşleme", "N-BaIoT, kalite, tekrar ve family_3 görevi",
             "Veri kalite ve tekrar raporları", "Veri tablosu", "Deney kayıtları mevcut"],
            ["4", "Önerilen Yöntem", "Üç mimari ve on varyantı açıklamak",
             "Model registry ve protokoller", "Mimari tablosu ve akış şeması", "Hazır"],
            ["5", "Deneysel Tasarım", "Seed, split, metrik ve ölçüm protokolü",
             "Stage32 summary ve stage33 manifest", "Deney matrisi", "Doğrulanmış"],
            ["6", "Sınıflandırma Bulguları", "Macro F1, MCC ve balanced accuracy",
             "Table 1", "Tablo 1 ve Şekil 3", "Hazır"],
            ["7", "İstatistiksel Analiz", "Friedman, Kendall W ve Wilcoxon-Holm",
             "Table 3 ve overall Wilcoxon CSV", "Tablo 3", "Hazır"],
            ["8", "Sınıf Bazlı Güvenlik Analizi", "Benign, Gafgyt ve Mirai FNR",
             "Table 4", "Tablo 4", "Hazır"],
            ["9", "Verimlilik ve Pareto", "Boyut, MAC, gecikme ve throughput ödünleşimi",
             "Table 2 ve Table 5", "Tablo 2, 5; Şekil 1, 2", "Hazır"],
            ["10", "Tartışma", "Model ailelerine göre sıkıştırma davranışını yorumlamak",
             "Stage32/33 sonuçları", "Önceki tablo ve şekiller", "İskelet hazır"],
            ["11", "Geçerlilik Tehditleri", "Tekrar, tek veri seti, n=5 ve host CPU sınırları",
             "Duplicate audit ve manifestler", "Yok", "Hazır"],
            ["12", "Sonuç", "Bilimsel ve mühendislik çıkarımlarını özetlemek",
             "Stage32/33 sonuçları", "Yok", "İskelet hazır"],
            ["13", "Tekrarlanabilirlik", "Kod, protokol, seed, manifest ve SHA-256",
             "Tüm manifestler", "Ek materyal tablosu", "Hazır"],
        ],
        columns=[
            "Bölüm No",
            "Bölüm",
            "Amaç",
            "Kanıt Kaynağı",
            "Tablo/Şekil",
            "Durum",
        ],
    )


def placement_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Tablo 1", "table_1_performance_summary.csv",
             "6. Sınıflandırma Bulguları",
             "Model ve sıkıştırma varyantlarına göre test performansı"],
            ["Tablo 2", "table_2_efficiency_summary.csv",
             "9. Verimlilik ve Pareto",
             "Model boyutu, MAC, host CPU gecikmesi ve throughput"],
            ["Tablo 3", "table_3_statistical_summary.csv",
             "7. İstatistiksel Analiz",
             "Macro F1 için Friedman ve Kendall W sonuçları"],
            ["Tablo 4", "table_4_classwise_fnr_summary.csv",
             "8. Sınıf Bazlı Güvenlik Analizi",
             "Sınıf bazlı yanlış negatif oranları"],
            ["Tablo 5", "table_5_pareto_optimal_solutions.csv",
             "9. Verimlilik ve Pareto",
             "Global ve model içi Pareto-optimal çözümler"],
            ["Şekil 1", "figure_1_macro_f1_vs_artifact.pdf",
             "9. Verimlilik ve Pareto",
             "Macro F1 ile artifact boyutu arasındaki ödünleşim"],
            ["Şekil 2", "figure_2_macro_f1_vs_latency.pdf",
             "9. Verimlilik ve Pareto",
             "Macro F1 ile host CPU gecikmesi arasındaki ödünleşim"],
            ["Şekil 3", "figure_3_macro_f1_heatmap.pdf",
             "6. Sınıflandırma Bulguları",
             "Model-varyant Macro F1 ısı haritası"],
        ],
        columns=["Öğe", "Dosya", "Makale Bölümü", "Başlık Taslağı"],
    )


def claims(head: dict[str, Any]) -> pd.DataFrame:
    a = head["best_accuracy"]
    s = head["smallest_artifact"]
    l = head["lowest_latency"]
    t = head["highest_throughput"]

    return pd.DataFrame(
        [
            [
                "C01",
                f"En yüksek ortalama Macro F1 {a['model']}/{a['variant']} "
                f"ile {a['value']:.6f} olarak elde edilmiştir.",
                "Table 1 ve Table 2",
                "Bu deney protokolünde elde edilmiştir.",
                "Evrensel olarak en iyi",
            ],
            [
                "C02",
                f"En küçük artifact {s['model']}/{s['variant']} için "
                f"{s['value']:.3f} KiB'dir.",
                "Table 2",
                "Ölçülen deployment artifact",
                "Tüm cihazlarda en küçük",
            ],
            [
                "C03",
                f"En düşük host CPU medyan gecikmesi {l['model']}/{l['variant']} "
                f"için {l['value']:.6f} ms'dir.",
                "Table 2",
                "Host CPU ortamında",
                "MCU gecikmesi",
            ],
            [
                "C04",
                f"En yüksek host CPU throughput {t['model']}/{t['variant']} "
                f"ile {t['value']:.3f} örnek/s'dir.",
                "Table 2",
                "Ölçülen host ortamında",
                "Gerçek cihaz throughput",
            ],
            [
                "C05",
                "Her üç modelde varyantların Macro F1 üzerindeki etkisi "
                "Friedman testine göre anlamlıdır.",
                "Table 3",
                "Omnibus varyant etkisi",
                "Bütün ikili farklar anlamlıdır",
            ],
            [
                "C06",
                "En yüksek doğruluk, en küçük artifact, en düşük gecikme ve "
                "en yüksek throughput farklı çözümlerde elde edilmiştir.",
                "Table 2 ve Table 5",
                "Mühendislik ödünleşimi vardır",
                "Tek evrensel en iyi model",
            ],
            [
                "C07",
                "Gecikme ve throughput sonuçları host CPU ölçümleridir.",
                "Stage33 manifest",
                "Host CPU profili",
                "MCU veya enerji ölçümü",
            ],
            [
                "C08",
                "Beş seed ikili Wilcoxon test gücünü sınırlar.",
                "Stage32 summary",
                "P-değeri etki büyüklüğüyle birlikte yorumlanmalıdır",
                "Anlamlı değilse fark yoktur",
            ],
            [
                "C09",
                "Yüksek tekrar oranı güçlü bir veri sızıntısı adayı ve "
                "geçerlilik tehdididir.",
                "Duplicate audit",
                "Tekrar/sızıntı adayı",
                "Kesin veri sızıntısı kanıtlandı",
            ],
            [
                "C10",
                "Gerçek MCU dağıtımı ve enerji ölçümü kapsam dışıdır.",
                "Stage32 summary ve Stage33 manifest",
                "Simülasyon ve host profilleme",
                "Gerçek cihazda doğrulandı",
            ],
        ],
        columns=[
            "İddia ID",
            "İddia",
            "Kanıt",
            "İzin Verilen Dil",
            "Kaçınılacak Dil",
        ],
    )


def manuscript(head: dict[str, Any]) -> str:
    a = head["best_accuracy"]
    s = head["smallest_artifact"]
    l = head["lowest_latency"]
    t = head["highest_throughput"]
    stats = head["friedman"]

    stat_lines = "\n".join(
        (
            f"- {model}: χ²({value['df']})={value['chi_square']:.6f}, "
            f"p={value['p']:.8f}, Kendall W={value['w']:.6f} "
            f"({value['magnitude']})."
        )
        for model, value in stats.items()
    )

    return f"""# TinyML-Based Intrusion Detection for Constrained IoT Devices:
# A Comparative Study of Quantized and Pruned Neural Architectures

## Türkçe Başlık

Kısıtlı IoT Cihazları İçin TinyML Tabanlı Saldırı Tespit Sistemi:
Kuantize Edilmiş ve Budanmış Yapay Sinir Ağı Mimarilerinin
Karşılaştırmalı İncelemesi

> Bu dosya Türkçe tam makale iskeletidir. Literatür atıfları henüz
> eklenmemiştir ve hiçbir kaynak uydurulmamıştır.

## Yazar Bilgileri

- Yazar: [Ad Soyad]
- Kurum: [Üniversite / Bölüm]
- E-posta: [E-posta]
- ORCID: [ORCID]

## Özet

[35. aşamada 150–250 kelimelik kaynaklı ve nihai özet yazılacaktır.]

Özetin zorunlu içeriği:

1. IoT IDS modellerinde doğruluk ile kaynak maliyeti arasındaki problem.
2. N-BaIoT family_3 üzerinde 3 model, 10 varyant, 5 seed ve 150 deney.
3. Fiziksel budama, DQ, PTQ, QAT ve budama+QAT karşılaştırması.
4. En yüksek Macro F1: {a['model']}/{a['variant']} = {a['value']:.6f}.
5. En küçük artifact: {s['model']}/{s['variant']} = {s['value']:.3f} KiB.
6. En düşük host gecikmesi: {l['model']}/{l['variant']} = {l['value']:.6f} ms.
7. Tek bir çözümün bütün hedeflerde üstün olmadığı sonucu.

## Anahtar Kelimeler

TinyML; intrusion detection system; IoT security; neural network
quantization; structured pruning; edge intelligence; N-BaIoT;
Pareto optimization

# I. GİRİŞ

## A. Problem Tanımı

- IoT cihazlarının bellek, işlem ve enerji kısıtlarını açıklayın.
- Bulut tabanlı IDS yaklaşımlarının gecikme, gizlilik ve bağlantı
  bağımlılığı sorunlarını tartışın.
- TinyML tabanlı yerel saldırı tespitinin önemini açıklayın.

## B. Araştırma Boşluğu

- Yalnızca doğruluk raporlayan çalışmaların dağıtım uygunluğunu eksik
  değerlendirdiğini literatürle gösterin.
- Budama ve kuantizasyonun aynı görev ve aynı seedler altında kapsamlı
  karşılaştırılmadığını araştırın.
- Sınıf bazlı FNR ve Pareto analizinin önemini vurgulayın.

## C. Katkılar

1. TinyML-MLP, Compact-DNN ve Tiny-1D-CNN'nin beş eşleştirilmiş seed ile
   karşılaştırılması.
2. P25/P50/P75 fiziksel budama, DQ, PTQ, QAT ve budama+QAT yöntemlerinin
   ortak protokol altında değerlendirilmesi.
3. Macro F1, MCC, dengeli doğruluk ve sınıf bazlı FNR'nin birlikte
   raporlanması.
4. Friedman, Kendall W, Wilcoxon-Holm ve rank-biserial etki
   büyüklüklerinin kullanılması.
5. Boyut, MAC, host CPU gecikmesi ve throughput ile Pareto analizi.
6. Seed, protokol, manifest ve SHA-256 kayıtlarıyla tekrarlanabilirlik.

# II. İLGİLİ ÇALIŞMALAR

> 35. aşamada güncel hakemli kaynaklarla yazılacaktır.

## A. IoT Saldırı Tespit Sistemleri

- Klasik makine öğrenmesi tabanlı IDS
- Derin öğrenme tabanlı IDS
- N-BaIoT kullanan çalışmalar

## B. TinyML ve Edge Güvenliği

- Kısıtlı cihazlarda TinyML
- Yerel güvenlik çıkarımı

## C. Budama ve Kuantizasyon

- Yapılandırılmış ve yapılandırılmamış budama
- DQ, PTQ ve QAT
- Budama+QAT

## D. Literatür Boşluğu

- Ortak seed ve ortak test protokolü eksikliği
- Güvenlik ve deployment metriklerinin birlikte incelenmemesi
- Pareto tabanlı seçim eksikliği

# III. VERİ SETİ VE ÖN İŞLEME

## A. N-BaIoT

- Toplam örnek: 7,062,606
- Özellik sayısı: 115
- Cihaz sayısı: 9
- Orijinal sınıf sayısı: 11
- family_3 sınıfları: benign, Gafgyt, Mirai

## B. Veri Kalitesi

- 89 CSV dosyasının tamamı sayısal ve sonlu.
- Bozuk dosya ve satır uyuşmazlığı yok.
- Sabit özellik yok.

## C. Tekrar Denetimi

- Benzersiz vektör: 2,482,676
- Fazla tekrar: 4,579,930
- Tekrar oranı: %64.8476
- Bu sonuç kesin veri sızıntısı değil, güçlü bir sızıntı adayı ve
  geçerlilik tehdidi olarak ifade edilmelidir.

## D. Ön İşleme

- Float32 kanonik veri
- Train verisinden öğrenilen StandardScaler
- Train-only sınıf ağırlıkları
- Ağırlıklı çapraz entropi
- Validation Macro F1 ile model seçimi
- Test setinin yalnızca nihai değerlendirmede kullanılması

# IV. ÖNERİLEN YÖNTEM

## A. TinyML-MLP

- 115→64→32→3
- 9,603 parametre
- 9,504 MAC

## B. Compact-DNN

- 115→128→64→32→3
- ReLU ve dropout=0.1
- 25,283 parametre
- 25,056 MAC

## C. Tiny-1D-CNN

- Üç 1D convolution katmanı ve adaptive pooling
- 1,699 parametre
- 58,816 MAC

## D. Varyantlar

- B0
- P25, P50, P75
- DQ, PTQ, QAT
- P25-QAT, P50-QAT, P75-QAT

## E. Yöntem Akışı

Veri denetimi → ön işleme → FP32 eğitim → budama/kuantizasyon →
validation seçimi → tek test → host CPU profilleme → istatistik → Pareto

# V. DENEYSEL TASARIM

## A. Deney Matrisi

- 3 model
- 10 varyant
- 5 seed: 42, 123, 2026, 3407, 8192
- 150 ana sonuç
- 450 sınıf bazlı sonuç

## B. Metrikler

- Birincil: Macro F1
- İkincil: MCC ve dengeli doğruluk
- Güvenlik: sınıf bazlı FNR
- Verimlilik: parametre, MAC, artifact, medyan/P95/P99 gecikme,
  throughput

## C. İstatistik

- Friedman
- Kendall W
- Eşleştirilmiş iki yönlü Wilcoxon
- Holm düzeltmesi
- Rank-biserial correlation
- Friedman kapı kontrolü

## D. Profilleme Kapsamı

- Tek CPU thread
- Batch=1 gecikme
- Batch=512 throughput
- Deterministik sentetik girdi
- Train/validation/test splitleri profillemede kullanılmadı
- MCU ölçümü yok

# VI. SINIFLANDIRMA BULGULARI

**Tablo 1 ve Şekil 3 burada kullanılacaktır.**

En yüksek ortalama Macro F1, {a['model']}/{a['variant']} ile
{a['value']:.6f} olarak elde edilmiştir.

- TinyML-MLP: QAT, P25-QAT ve P25 üst sıralardadır.
- Compact-DNN: QAT doğruluk lideridir; P25-QAT yakın alternatiftir.
- Tiny-1D-CNN: QAT ortalamayı yükseltmiş, değişkenlik daha yüksektir.

# VII. İSTATİSTİKSEL ANALİZ

**Tablo 3 burada kullanılacaktır.**

{stat_lines}

Beş seed nedeniyle p-değerleri, etki büyüklükleri ve eşleştirilmiş
farklarla birlikte yorumlanmalıdır.

# VIII. SINIF BAZLI GÜVENLİK ANALİZİ

**Tablo 4 burada kullanılacaktır.**

Benign, Gafgyt ve Mirai FNR sonuçları ayrı değerlendirilmelidir.
Macro F1 tek başına saldırı sınıflarındaki yanlış negatif riskini
tam olarak açıklamayabilir.

# IX. VERİMLİLİK VE PARETO ANALİZİ

**Tablo 2, Tablo 5, Şekil 1 ve Şekil 2 burada kullanılacaktır.**

- En küçük artifact: {s['model']}/{s['variant']} =
  {s['value']:.3f} KiB.
- En düşük host gecikmesi: {l['model']}/{l['variant']} =
  {l['value']:.6f} ms.
- En yüksek host throughput: {t['model']}/{t['variant']} =
  {t['value']:.3f} örnek/s.
- Global Pareto çözüm sayısı: {head['global_pareto_count']}.

Tek bir bileşik skor yerine doğruluk, boyut, gecikme ve MAC ayrı
amaçlar olarak değerlendirilmelidir.

# X. TARTIŞMA

## A. QAT Davranışı

- Eğitim sırasında quantization hatasına adaptasyon
- MLP ve DNN katmanlarının int8 dönüşümüne uygunluğu
- PTQ ile karşılaştırmalı yorum

## B. Budama Düzeyi

- P25'in performans-kaynak dengesi
- P50/P75'te kapasite kaybı riski
- Model ailesine bağlı duyarlılık

## C. CNN Değişkenliği

- Az parametreye karşın yüksek MAC
- Seed duyarlılığı
- Ortalama, SS ve FNR'nin birlikte yorumlanması

## D. Dağıtım Kararı

- Doğruluk: {a['model']}/{a['variant']}
- Minimum artifact: {s['model']}/{s['variant']}
- Düşük host gecikmesi: {l['model']}/{l['variant']}
- Yüksek host throughput: {t['model']}/{t['variant']}
- Nihai seçim Pareto ve sınıf bazlı FNR ile yapılmalıdır.

# XI. GEÇERLİLİK TEHDİTLERİ

## A. İç Geçerlilik

- Yüksek veri tekrar oranı
- Split sınırlarında sızıntı adayı
- Runtime ve CPU'ya bağlı profilleme

## B. Dış Geçerlilik

- Tek veri seti
- Farklı IoT ortamlarına sınırlı genellenebilirlik
- Gerçek MCU doğrulaması yok

## C. Sonuç Geçerliliği

- Beş seed
- Holm düzeltmesi
- İstatistiksel ve mühendislik öneminin ayrılması

## D. Yapısal Geçerlilik

- Macro F1 ana metrik
- Sınıf bazlı FNR güvenlik tamamlayıcısı
- Enerji ölçümü kapsam dışında

# XII. SONUÇ

Bu çalışma üç hafif sinir ağı mimarisini on FP32, budama,
kuantizasyon ve budama+QAT varyantı altında karşılaştırmıştır.
En yüksek ortalama Macro F1 {a['model']}/{a['variant']} ile
{a['value']:.6f} olarak elde edilmiştir. En küçük artifact, en düşük
host gecikmesi ve en yüksek throughput farklı çözümlerde görülmüştür.
Bu nedenle kısıtlı IoT dağıtımlarında tek bir evrensel en iyi model
yerine hedef cihaz gereksinimlerine göre Pareto tabanlı seçim
yapılmalıdır.

Gelecek çalışmalarda gerçek MCU üzerinde RAM, gecikme ve enerji
ölçümleri; farklı veri setlerinde dış doğrulama; tekrar-duyarlı bölme
ve çevrim içi saldırı tespiti incelenmelidir.

# XIII. TEKRARLANABİLİRLİK

- Python ve PyTorch kodları
- Sabit seedler
- Protokol JSON dosyaları
- SHA-256 ve manifestler
- Ham ve toplulaştırılmış CSV sonuçları
- Host CPU ortam kaydı

# EKLER

- Ek A: Model hiperparametreleri
- Ek B: Tam Wilcoxon-Holm sonuçları
- Ek C: Tam sınıf bazlı FNR sonuçları
- Ek D: Yazılım ve donanım ortamı
- Ek E: Artifact ve manifest dosyaları

# 35. AŞAMAYA AKTARILACAK İŞLER

1. Güncel hakemli literatür taraması
2. İlgili çalışmalar karşılaştırma tablosu
3. İngilizce IEEE Access özeti
4. Kaynaklı giriş ve ilgili çalışmalar
5. FNR sonuçlarının ayrıntılı güvenlik yorumu
6. Seçilmiş Wilcoxon-Holm çiftlerinin metne aktarılması
7. Yöntem akış şeması
8. IEEE Access biçim düzenlemesi
"""


def main() -> None:
    args = parse_args()
    report_dir = args.report_directory.resolve()
    stage33_dir = args.stage33_directory.resolve()
    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "outline": output_dir / "manuscript_outline_tr.md",
        "section_map": output_dir / "section_evidence_map.csv",
        "placement": output_dir / "table_figure_placement_plan.csv",
        "claims": output_dir / "manuscript_claims_checklist.csv",
        "manifest": output_dir / "stage34_manifest.json",
    }

    if any(path.exists() for path in outputs.values()) and not args.overwrite:
        raise FileExistsError(
            "34. aşama çıktıları mevcut. --overwrite kullan."
        )

    print("=" * 78)
    print("34. Aşama — IEEE Access Makale İskeleti ve Kanıt Haritası")
    print("=" * 78)
    print(f"Rapor klasörü : {report_dir}")
    print(f"33. aşama     : {stage33_dir}")
    print(f"Çıktı klasörü : {output_dir}")
    print("=" * 78)

    inputs = load_inputs(report_dir, stage33_dir)
    head = headline(inputs)

    section_frame = section_map()
    placement_frame = placement_plan()
    claims_frame = claims(head)
    outline_text = manuscript(head)

    write_text(outputs["outline"], outline_text)
    write_csv(outputs["section_map"], section_frame)
    write_csv(outputs["placement"], placement_frame)
    write_csv(outputs["claims"], claims_frame)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 34,
        "target_venue": "IEEE Access",
        "language": "Turkish",
        "input_artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in inputs["paths"].items()
        },
        "generated_artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
            if name != "manifest"
        },
        "headline_results": head,
        "row_counts": {
            "section_evidence_map": len(section_frame),
            "table_figure_placement_plan": len(placement_frame),
            "manuscript_claims_checklist": len(claims_frame),
        },
        "new_training_performed": False,
        "test_re_evaluated": False,
        "new_statistical_test_performed": False,
        "literature_citations_generated": False,
        "mcu_measurement_claimed": False,
        "validation_passed": True,
    }

    write_json(outputs["manifest"], manifest)

    print()
    print("=" * 78)
    print("34. Aşama Tamamlandı")
    print("=" * 78)
    print("Makale iskeleti       : 1 Markdown")
    print(f"Bölüm-kanıt haritası  : {len(section_frame)} satır")
    print(f"Tablo/şekil planı     : {len(placement_frame)} satır")
    print(f"İddia kontrol listesi : {len(claims_frame)} satır")
    print("Yeni eğitim           : False")
    print("Test tekrarlandı      : False")
    print("Yeni istatistik       : False")
    print("Uydurma atıf          : False")
    print("MCU ölçümü iddiası    : False")
    print("Doğrulama geçti       : True")
    print(f"Çıktı klasörü         : {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
