# TinyML Tabanlı IoT Saldırı Tespit Sistemi

## Proje Başlığı

**Kısıtlı IoT Cihazları İçin TinyML Tabanlı Saldırı Tespit Sistemi: Kuantize Edilmiş ve Budanmış Yapay Sinir Ağı Mimarilerinin Simülasyon Tabanlı Karşılaştırmalı İncelemesi**

### İngilizce Başlık

**Simulation-Based Comparative Analysis of Quantized and Pruned TinyML Neural Architectures for Intrusion Detection in Resource-Constrained IoT Devices**

---

## 1. Projenin Amacı

Bu proje, kısıtlı IoT cihazlarına yönelik hafif yapay sinir ağı tabanlı saldırı tespit modellerinin kontrollü bir yazılım simülasyonu ortamında geliştirilmesini ve karşılaştırılmasını amaçlamaktadır.

Çalışmada farklı hafif sinir ağı mimarileri aşağıdaki model sıkıştırma yöntemleriyle değerlendirilecektir:

- Yapılandırılmış budama
- Yapılandırılmamış budama
- Dinamik INT8 kuantizasyon
- Post-training quantization
- Quantization-aware training
- Budama ve kuantizasyonun birlikte uygulanması

Modeller yalnızca sınıflandırma doğruluğuna göre değil; güvenlik performansı, model karmaşıklığı, model boyutu, tahmini bellek gereksinimi ve kontrollü CPU çıkarım gecikmesi açısından karşılaştırılacaktır.

---

## 2. Çalışmanın Kapsamı

Bu çalışma tamamen yazılım ve simülasyon tabanlıdır.

Fiziksel mikrodenetleyici veya gerçek IoT donanımı kullanılmayacaktır.

Çalışmada değerlendirilecek ölçütler:

### Sınıflandırma Performansı

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Matthews Correlation Coefficient
- ROC-AUC
- PR-AUC
- False Positive Rate
- False Negative Rate
- Sınıf bazlı precision, recall ve F1-score
- Confusion Matrix

### Model Karmaşıklığı

- Toplam parametre sayısı
- Eğitilebilir parametre sayısı
- Sıfır olmayan parametre sayısı
- Model sıfırlık oranı
- MAC sayısı
- FLOP sayısı
- Model dosya boyutu
- Sıkıştırma oranı
- Tahmini ağırlık belleği
- Tahmini aktivasyon belleği

### Simülasyon Tabanlı Performans

- Ortalama CPU çıkarım gecikmesi
- Medyan çıkarım gecikmesi
- p90, p95 ve p99 gecikme
- Throughput
- Pik süreç belleği
- Temsili cihaz profillerine teorik uygunluk

---

## 3. Akademik Sınırlar

Bu çalışma kapsamında aşağıdaki iddialarda bulunulmayacaktır:

- Gerçek mikrodenetleyici gecikmesi ölçülmüştür.
- Gerçek enerji tüketimi belirlenmiştir.
- Gerçek RAM veya Flash kullanımı doğrulanmıştır.
- Sistem fiziksel cihaz üzerinde gerçek zamanlı çalışmaktadır.
- Model fiziksel IoT ortamında deployment-ready durumdadır.

Masaüstü bilgisayar üzerinde ölçülen gecikme değerleri yalnızca kontrollü yazılım ortamındaki göreli karşılaştırmalar için kullanılacaktır.

---

## 4. Kullanılacak Veri Seti

Birincil veri seti olarak N-BaIoT veri setinin kullanılması planlanmaktadır.

Veri seti kullanılmadan önce aşağıdaki kontroller gerçekleştirilecektir:

- Dosya bütünlüğü
- SHA-256 hash değerleri
- Sınıf dağılımı
- Eksik değerler
- Sonsuz değerler
- Tekrarlanan kayıtlar
- Kaynak cihaz bilgileri
- Kaynak dosya bilgileri
- Saldırı türleri
- Veri sızıntısı riskleri

Ham veri seti Git deposuna eklenmeyecektir.

---

## 5. Planlanan Model Mimarileri

Çalışmada üç temel hafif sinir ağı mimarisi değerlendirilecektir:

1. TinyML-MLP
2. Compact-DNN
3. Tiny-1D-CNN

Bütün mimariler aynı veri bölmesi, ön işleme protokolü, değerlendirme metrikleri ve random seed değerleriyle karşılaştırılacaktır.

---

## 6. Model Varyantları

Her temel mimari için aşağıdaki model varyantlarının değerlendirilmesi planlanmaktadır:

| Kod | Model varyantı |
|---|---|
| B0 | FP32 temel model |
| P25 | %25 budanmış model |
| P50 | %50 budanmış model |
| P75 | %75 budanmış model |
| DQ | Dinamik INT8 kuantize model |
| PTQ | Post-training INT8 kuantize model |
| QAT | Quantization-aware training modeli |
| P25-QAT | %25 budama ve QAT |
| P50-QAT | %50 budama ve QAT |
| P75-QAT | %75 budama ve QAT |

---

## 7. Deney Tekrarları

Temel deneyler aşağıdaki random seed değerleriyle tekrarlanacaktır:

```text
42
123
2026
3407
8192