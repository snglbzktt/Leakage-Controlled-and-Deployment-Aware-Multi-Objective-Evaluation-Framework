# Araştırma Soruları ve Hipotezler

## Çalışmanın Başlığı

**Kısıtlı IoT Cihazları İçin TinyML Tabanlı Saldırı Tespit Sistemi:
Kuantize Edilmiş ve Budanmış Yapay Sinir Ağı Mimarilerinin
Simülasyon Tabanlı Karşılaştırmalı İncelemesi**

---

## 1. Belgenin Amacı

Bu belge, çalışmada cevaplanacak araştırma sorularını, test edilecek
hipotezleri, temel değerlendirme ölçütlerini ve bilimsel karar
kurallarını tanımlamaktadır.

Araştırma soruları deneyler başlamadan önce belirlenmiştir. Böylece
sonuçlar elde edildikten sonra araştırma hedeflerinin değiştirilmesi veya
yalnızca olumlu sonuçların seçilmesi önlenmeye çalışılacaktır.

Çalışma yalnızca yazılım tabanlı simülasyon ortamında yürütülecektir.
Gerçek mikrodenetleyici gecikmesi, gerçek enerji tüketimi veya gerçek
donanım performansı ölçüldüğü iddia edilmeyecektir.

---

## 2. Temel Araştırma Sorusu

### AS0 — Genel Araştırma Sorusu

Kontrollü bir simülasyon ortamında hangi hafif sinir ağı mimarisi,
budama oranı ve kuantizasyon yöntemi; saldırı tespit performansı,
model boyutu, hesaplama karmaşıklığı ve tahmini kaynak gereksinimi
arasında en uygun dengeyi sağlamaktadır?

Bu soru çalışmanın ana araştırma sorusudur.

---

## 3. Alt Araştırma Soruları

### AS1 — Mimari Seçiminin Etkisi

Farklı hafif sinir ağı mimarileri, aynı veri hazırlama ve eğitim
koşulları altında saldırı tespit performansı ve kaynak gereksinimi
bakımından anlamlı biçimde farklılaşmakta mıdır?

Karşılaştırılması planlanan mimariler:

- Hafif çok katmanlı algılayıcı
- Kompakt derin çok katmanlı ağ
- Bir boyutlu evrişimsel sinir ağı
- Uygun olması durumunda kompakt residual ağ

#### H0-1

Mimariler arasında Macro F1-score, sınıf bazlı recall, model boyutu ve
hesaplama karmaşıklığı açısından istatistiksel olarak anlamlı fark yoktur.

#### H1-1

En az bir mimari, saldırı tespit performansı veya kaynak verimliliği
açısından diğer mimarilerden anlamlı biçimde farklıdır.

---

### AS2 — Budama Oranının Etkisi

Budama oranı arttıkça modelin kaynak gereksinimi ve saldırı tespit
performansı nasıl değişmektedir?

Değerlendirilecek temel budama oranları:

- %0
- %25
- %50
- %75

#### H0-2

Budama oranı, model performansı ve kaynak gereksinimi üzerinde anlamlı
bir etkiye sahip değildir.

#### H1-2

Budama oranı arttıkça sıfır olmayan parametre sayısı ve modelin teorik
hesaplama yükü azalırken, belirli bir eşik sonrasında saldırı tespit
performansı anlamlı biçimde düşmektedir.

#### Temel ölçütler

- Macro F1-score
- Sınıf bazlı recall
- False Negative Rate
- Sıfır olmayan parametre sayısı
- Sıfırlık oranı
- Model dosya boyutu
- MAC ve FLOP değerleri
- Yazılım tabanlı çıkarım gecikmesi

---

### AS3 — Budama Sonrası Fine-Tuning Etkisi

Budama sonrası gerçekleştirilen fine-tuning işlemi, budamanın neden
olduğu performans kaybını ne ölçüde geri kazandırmaktadır?

#### H0-3

Budama sonrası fine-tuning, budanmış modelin performansında anlamlı
bir iyileşme sağlamaz.

#### H1-3

Budama sonrası fine-tuning, budanmış modelin Macro F1-score ve sınıf
bazlı recall değerlerinde anlamlı iyileşme sağlar.

#### Karşılaştırılacak durumlar

1. FP32 temel model
2. Fine-tuning uygulanmamış budanmış model
3. Fine-tuning uygulanmış budanmış model

---

### AS4 — Kuantizasyon Yöntemlerinin Etkisi

Farklı kuantizasyon yöntemleri saldırı tespit performansı, model boyutu
ve yazılım tabanlı çıkarım verimliliği açısından nasıl farklılaşmaktadır?

Değerlendirilmesi planlanan yöntemler:

- FP32 temel model
- Dinamik kuantizasyon
- INT8 post-training quantization
- Quantization-aware training

#### H0-4

Kuantizasyon yöntemleri arasında performans ve kaynak verimliliği
açısından anlamlı fark yoktur.

#### H1-4

INT8 kuantizasyon model boyutunu ve ağırlık bellek ihtiyacını azaltırken,
quantization-aware training post-training quantization yöntemine göre
daha düşük sınıflandırma kaybı sağlamaktadır.

#### Temel ölçütler

- Macro F1-score
- Weighted F1-score
- Sınıf bazlı recall
- False Negative Rate
- Model dosya boyutu
- Sıkıştırma oranı
- Ortalama çıkarım gecikmesi
- p95 çıkarım gecikmesi
- Throughput

---

### AS5 — Budama ve Kuantizasyonun Birleşik Etkisi

Budama ve kuantizasyon birlikte uygulandığında, yöntemlerin tek başına
uygulanmasına kıyasla daha iyi bir doğruluk–verimlilik dengesi elde
edilebilir mi?

#### H0-5

Budama ve kuantizasyonun birlikte uygulanması, tek başına uygulanan
yöntemlere göre anlamlı bir avantaj sağlamaz.

#### H1-5

Budama ve kuantizasyonun birlikte uygulanması, kabul edilebilir bir
performans kaybı karşılığında model boyutu ve kaynak gereksiniminde
daha yüksek azalma sağlar.

#### Karşılaştırma grupları

1. FP32 temel model
2. Yalnızca budanmış model
3. Yalnızca kuantize edilmiş model
4. Budanmış ve kuantize edilmiş model

---

### AS6 — Saldırı Sınıfları Üzerindeki Etki

Model sıkıştırma yöntemleri tüm saldırı sınıflarını aynı düzeyde mi
etkilemektedir?

#### H0-6

Budama ve kuantizasyonun sınıf bazlı recall ve False Negative Rate
üzerindeki etkisi saldırı sınıfları arasında farklılaşmamaktadır.

#### H1-6

Bazı saldırı sınıfları, budama ve kuantizasyon işlemlerinden diğer
sınıflara göre daha fazla etkilenmektedir.

#### İncelenecek değerler

- Sınıf bazlı precision
- Sınıf bazlı recall
- Sınıf bazlı F1-score
- False Negative Rate
- Confusion Matrix
- Sıkıştırma öncesi ve sonrası hata dağılımı

Bu araştırma sorusu güvenlik açısından kritik öneme sahiptir. Genel
doğruluk değerindeki küçük bir değişim, belirli bir saldırı sınıfındaki
yüksek yanlış negatif oranını gizleyebilir.

---

### AS7 — Çoklu Seed Kararlılığı

Modellerin performansı farklı random seed değerleri altında ne ölçüde
kararlıdır?

#### H0-7

Model türü ve sıkıştırma yöntemi, deney sonuçlarının random seed
değişkenliğini etkilemez.

#### H1-7

Bazı mimari ve sıkıştırma yapılandırmaları farklı random seed değerleri
altında daha yüksek performans değişkenliği göstermektedir.

#### Raporlanacak değerler

- Ortalama
- Standart sapma
- Medyan
- Minimum
- Maksimum
- %95 güven aralığı

Temel deneyler en az beş farklı random seed ile yürütülecektir.

Planlanan başlangıç seed değerleri:

- 42
- 123
- 2026
- 3407
- 8192

Seed değerleri deney protokolü kesinleştirildikten sonra değiştirilebilir;
ancak deney başladıktan sonra sonuçlara göre seçilmeyecektir.

---

### AS8 — Model Karmaşıklığı ve Gecikme İlişkisi

Parametre sayısı, MAC/FLOP değeri ve model dosya boyutu ile kontrollü
CPU ortamındaki çıkarım gecikmesi arasında nasıl bir ilişki bulunmaktadır?

#### H0-8

Model karmaşıklığı göstergeleri ile yazılım tabanlı çıkarım gecikmesi
arasında anlamlı ilişki yoktur.

#### H1-8

Parametre sayısı, MAC/FLOP değeri ve model boyutu ile çıkarım gecikmesi
arasında anlamlı ilişki bulunmaktadır.

Bu analizde masaüstü CPU gecikmesinin mikrodenetleyici gecikmesini
temsil etmediği açıkça belirtilecektir.

---

### AS9 — Hedef Cihaz Profillerine Teorik Uygunluk

Hangi model yapılandırmaları önceden tanımlanan RAM ve Flash bütçelerine
teorik olarak uyum sağlamaktadır?

#### H0-9

Model sıkıştırma yöntemleri, modellerin hedef kaynak profillerine uyum
durumunu değiştirmez.

#### H1-9

Budama ve kuantizasyon, bazı model yapılandırmalarının daha düşük RAM ve
Flash bütçelerine teorik olarak uyum sağlamasına olanak verir.

#### Değerlendirme yaklaşımı

Her hedef cihaz profili için aşağıdaki sınırlar tanımlanacaktır:

- Maksimum model boyutu
- Maksimum ağırlık belleği
- Maksimum tahmini aktivasyon belleği
- Maksimum hesaplama bütçesi

Bu analiz teorik dağıtım uygunluğu olarak adlandırılacaktır. Gerçek cihaz
uyumluluğu veya gerçek deployment sonucu olarak sunulmayacaktır.

---

### AS10 — Açıklanabilirlik ve Özellik Önem Kararlılığı

Model sıkıştırma sonrasında modelin karar vermek için kullandığı temel
özellikler değişmekte midir?

#### H0-10

FP32, budanmış ve kuantize edilmiş modellerin özellik önem sıralamaları
arasında anlamlı fark yoktur.

#### H1-10

Budama veya kuantizasyon işlemi, modelin özellik önem dağılımını ve
karar mekanizmasını değiştirir.

#### Kullanılması planlanan yöntemler

- Permutation Feature Importance
- SHAP
- Uygun olması durumunda gradient tabanlı önem yöntemleri

Açıklanabilirlik analizi yalnızca seçilen nihai veya Pareto-optimal
modeller üzerinde uygulanabilir.

---

## 4. Birincil ve İkincil Sonuç Ölçütleri

### 4.1. Birincil Güvenlik Ölçütü

Çalışmanın temel sınıflandırma metriği:

**Macro F1-score**

Macro F1-score, her sınıfa eşit önem verdiği için dengesiz saldırı
verilerinde Accuracy değerinden daha güvenilir bir genel ölçüttür.

### 4.2. Birincil Risk Ölçütü

Güvenlik açısından temel risk ölçütü:

**Sınıf bazlı False Negative Rate**

Bir saldırı örneğinin normal veya yanlış saldırı sınıfı olarak
değerlendirilmesi güvenlik açısından kritik kabul edilecektir.

### 4.3. Birincil Verimlilik Ölçütleri

- Model dosya boyutu
- Sıfır olmayan parametre sayısı
- MAC sayısı
- Tahmini ağırlık belleği
- Kontrollü CPU ortamında ortalama çıkarım gecikmesi

### 4.4. İkincil Sınıflandırma Ölçütleri

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Weighted F1-score
- Matthews Correlation Coefficient
- ROC-AUC
- PR-AUC
- False Positive Rate
- Confusion Matrix

---

## 5. Deneysel Karar Kuralları

Bir model yalnızca en yüksek Accuracy değerine sahip olduğu için en iyi
model olarak seçilmeyecektir.

Model seçiminde aşağıdaki ölçütler birlikte değerlendirilecektir:

1. Macro F1-score
2. Sınıf bazlı recall
3. False Negative Rate
4. Model dosya boyutu
5. Parametre ve sıfır olmayan parametre sayısı
6. MAC/FLOP değeri
7. Tahmini bellek gereksinimi
8. Kontrollü yazılım gecikmesi
9. Farklı seed değerleri altındaki kararlılık

Tek bir genel kazanan ilan etmek yerine Pareto-optimal modeller
belirlenecektir.

---

## 6. İstatistiksel Analiz İlkeleri

Deney sonuçları yalnızca tek bir çalıştırmaya dayandırılmayacaktır.

Her temel yapılandırma için en az beş tekrar yapılacaktır.

Sonuçlar aşağıdaki biçimde sunulacaktır:

- Ortalama
- Standart sapma
- Medyan
- %95 güven aralığı
- Etki büyüklüğü

İstatistiksel test seçimi veri dağılımı ve deney yapısına göre
belirlenecektir.

Kullanılabilecek yöntemler:

- Paired t-test
- Wilcoxon signed-rank testi
- Friedman testi
- Post-hoc testler
- Holm veya Benjamini–Hochberg düzeltmesi
- Cohen's d
- Rank-biserial correlation
- Kendall's W

Yalnızca p-value raporlanmayacaktır. Etki büyüklüğü ve pratik önem de
yorumlanacaktır.

---

## 7. Simülasyon Kapsamı ve Sınırlılık

Bu çalışma fiziksel bir mikrodenetleyici üzerinde yürütülmeyecektir.

Bu nedenle aşağıdaki sonuçlar üretilebilir:

- Yazılım tabanlı gecikme
- Kontrollü CPU throughput değeri
- Model dosya boyutu
- Parametre sayısı
- MAC/FLOP değeri
- Tahmini ağırlık belleği
- Tahmini aktivasyon belleği
- Hedef kaynak profillerine teorik uyumluluk

Aşağıdaki sonuçlar üretildiği iddia edilmeyecektir:

- Gerçek mikrodenetleyici gecikmesi
- Gerçek enerji tüketimi
- Gerçek Flash kullanım doğrulaması
- Gerçek RAM kullanım doğrulaması
- Gerçek zamanlı deployment garantisi
- Fiziksel cihaz üzerindeki kararlılık

---

## 8. Araştırmanın Başarı Koşulu

Araştırma, aşağıdaki soruya kanıta dayalı bir cevap üretebildiğinde
başarılı kabul edilecektir:

> Hangi mimari ve sıkıştırma yapılandırmaları saldırı tespit
> güvenilirliğini kabul edilebilir düzeyde korurken model boyutu,
> hesaplama karmaşıklığı ve tahmini kaynak gereksinimini en fazla
> azaltmaktadır?

Bu cevap tek bir metrik üzerinden değil; güvenlik başarısı, kaynak
verimliliği, istatistiksel kararlılık ve sınıf bazlı performans birlikte
değerlendirilerek verilecektir.