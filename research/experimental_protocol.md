# Deneysel Araştırma Protokolü

## Çalışmanın Başlığı

**Kısıtlı IoT Cihazları İçin TinyML Tabanlı Saldırı Tespit Sistemi: Kuantize Edilmiş ve Budanmış Yapay Sinir Ağı Mimarilerinin Simülasyon Tabanlı Karşılaştırmalı İncelemesi**

### İngilizce Başlık

**Simulation-Based Comparative Analysis of Quantized and Pruned TinyML Neural Architectures for Intrusion Detection in Resource-Constrained IoT Devices**

---

## 1. Protokolün Amacı

Bu belge; çalışmada kullanılacak veri setini, veri hazırlama yöntemlerini, model mimarilerini, eğitim koşullarını, budama ve kuantizasyon yöntemlerini, deney tekrarlarını, değerlendirme metriklerini, istatistiksel analizleri ve sonuçların raporlanma kurallarını deneyler başlamadan önce tanımlamaktadır.

Deneysel protokolün önceden belirlenmesinin temel amaçları şunlardır:

1. Deney sonuçlarına göre yöntem değiştirilmesini önlemek
2. Veri sızıntısı risklerini azaltmak
3. Modeller arasında adil karşılaştırma sağlamak
4. Deneylerin tekrar üretilebilir olmasını sağlamak
5. Yalnızca olumlu sonuçların seçilmesini önlemek
6. İstatistiksel analizlerin önceden belirlenmesini sağlamak
7. Araştırma iddialarını deneysel kanıtlarla sınırlandırmak

Bu çalışma yalnızca yazılım tabanlı simülasyon ortamında yürütülecektir. Fiziksel bir mikrodenetleyici veya gerçek IoT cihazı kullanılmayacaktır.

---

## 2. Araştırmanın Deneysel Kapsamı

Çalışmada aşağıdaki model türleri karşılaştırılacaktır:

1. FP32 temel sinir ağı modelleri
2. Budanmış sinir ağı modelleri
3. Kuantize edilmiş sinir ağı modelleri
4. Budama ve kuantizasyonun birlikte uygulandığı modeller

Modeller aşağıdaki iki ana boyutta değerlendirilecektir:

### 2.1. Güvenlik ve Sınıflandırma Performansı

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
- Sınıf bazlı precision
- Sınıf bazlı recall
- Sınıf bazlı F1-score
- Confusion Matrix

### 2.2. Simülasyon Tabanlı Kaynak Verimliliği

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
- Kontrollü CPU ortamında çıkarım gecikmesi
- Kontrollü CPU ortamında throughput
- Pik süreç belleği
- Hedef kaynak profillerine teorik uygunluk

---

## 3. Veri Seti

### 3.1. Birincil Veri Seti

Çalışmanın birincil veri seti olarak **N-BaIoT** veri setinin kullanılması planlanmaktadır.

Veri setinin aşağıdaki özellikleri kayıt altına alınacaktır:

- Veri setinin resmi kaynağı
- İndirme tarihi
- Dosya adları
- Dosya boyutları
- Dosya hash değerleri
- Cihaz türleri
- Normal trafik örnekleri
- Saldırı türleri
- Her sınıftaki örnek sayısı
- Toplam özellik sayısı
- Eksik değer sayısı
- Sonsuz değer sayısı
- Tekrarlanan örnek sayısı
- Sınıf dağılımı
- Kaynak dosya ve cihaz bilgileri

Veri seti üzerinde işlem yapılmadan önce ham dosyaların SHA-256 hash değerleri hesaplanacaktır. Böylece kullanılan veri sürümü doğrulanabilir olacaktır.

### 3.2. Birincil Sınıflandırma Görevi

Çalışmanın birincil görevi çok sınıflı saldırı tespitidir.

Modelin normal trafik ile farklı saldırı türlerini birbirinden ayırması beklenecektir.

### 3.3. İkincil Sınıflandırma Görevi

İkincil deney olarak ikili sınıflandırma gerçekleştirilebilir:

- Normal
- Saldırı

İkili sınıflandırma sonuçları ana katkı olarak değil, çok sınıflı sonuçları tamamlayan ikincil analiz olarak raporlanacaktır.

---

## 4. Veri Sızıntısının Önlenmesi

Veri sızıntısının engellenmesi çalışmanın temel metodolojik koşullarından biridir.

Rastgele satır bazlı veri bölme işlemi, aynı cihazdan veya aynı saldırı oturumundan gelen çok benzer örneklerin eğitim ve test kümelerine dağılmasına neden olabilir. Bu durum performansın olduğundan yüksek görünmesine yol açabilir.

Bu nedenle veri bölme işleminde mümkün olan en güçlü grup bilgisi kullanılacaktır.

Grup değişkenleri öncelik sırasına göre şunlardır:

1. Kaynak cihaz
2. Kaynak dosya
3. Saldırı oturumu
4. Trafik senaryosu
5. Aynı özellik vektörüne sahip örnek grubu

Aynı gruba ait örnekler birden fazla veri bölümünde bulunmayacaktır.

### 4.1. Tekrarlanan Örnekler

Tamamen aynı özellik değerlerine sahip satırlar tespit edilecektir.

Aşağıdaki iki yöntemden biri kullanılacaktır:

1. Tekrarlanan satırların veri bölme işleminden önce kaldırılması
2. Aynı hash değerine sahip satırların tek bir grup olarak değerlendirilmesi

Kullanılan yöntem ve kaldırılan örnek sayısı raporlanacaktır.

### 4.2. Ön İşleme Sızıntısı

Aşağıdaki işlemler yalnızca eğitim verisi kullanılarak öğrenilecektir:

- Eksik değer doldurma
- Standartlaştırma
- Normalizasyon
- Özellik seçimi
- Varyans tabanlı filtreleme
- Sınıf ağırlıklarının hesaplanması
- Kuantizasyon kalibrasyonu

Doğrulama ve test kümeleri üzerinde `fit` işlemi yapılmayacaktır.

---

## 5. Veri Bölme Protokolü

### 5.1. Birincil Veri Bölmesi

Birincil deneylerde aşağıdaki oranlar kullanılacaktır:

- Eğitim: %70
- Doğrulama: %15
- Test: %15

Bölme işlemi sınıf dağılımını mümkün olduğunca koruyan grup tabanlı bir yöntemle gerçekleştirilecektir.

Aynı cihaz, kaynak dosya veya saldırı oturumuna ait örneklerin farklı bölümlere dağılması engellenecektir.

### 5.2. Veri Bölmesinin Dondurulması

Oluşturulan train, validation ve test indeksleri dosyaya kaydedilecektir.

Örnek dosyalar:

```text
data/splits/train_indices.csv
data/splits/validation_indices.csv
data/splits/test_indices.csv
data/splits/split_metadata.json