# TinyML-Based Intrusion Detection for Constrained IoT Devices:
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
4. En yüksek Macro F1: Compact-DNN/QAT = 0.991846.
5. En küçük artifact: Tiny-1D-CNN/P75 = 4.200 KiB.
6. En düşük host gecikmesi: TinyML-MLP/PTQ = 0.018750 ms.
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

En yüksek ortalama Macro F1, Compact-DNN/QAT ile
0.991846 olarak elde edilmiştir.

- TinyML-MLP: QAT, P25-QAT ve P25 üst sıralardadır.
- Compact-DNN: QAT doğruluk lideridir; P25-QAT yakın alternatiftir.
- Tiny-1D-CNN: QAT ortalamayı yükseltmiş, değişkenlik daha yüksektir.

# VII. İSTATİSTİKSEL ANALİZ

**Tablo 3 burada kullanılacaktır.**

- TinyML-MLP: χ²(9)=41.552727, p=0.00000397, Kendall W=0.923394 (large).
- Compact-DNN: χ²(9)=37.538182, p=0.00002109, Kendall W=0.834182 (large).
- Tiny-1D-CNN: χ²(9)=40.063107, p=0.00000740, Kendall W=0.890291 (large).

Beş seed nedeniyle p-değerleri, etki büyüklükleri ve eşleştirilmiş
farklarla birlikte yorumlanmalıdır.

# VIII. SINIF BAZLI GÜVENLİK ANALİZİ

**Tablo 4 burada kullanılacaktır.**

Benign, Gafgyt ve Mirai FNR sonuçları ayrı değerlendirilmelidir.
Macro F1 tek başına saldırı sınıflarındaki yanlış negatif riskini
tam olarak açıklamayabilir.

# IX. VERİMLİLİK VE PARETO ANALİZİ

**Tablo 2, Tablo 5, Şekil 1 ve Şekil 2 burada kullanılacaktır.**

- En küçük artifact: Tiny-1D-CNN/P75 =
  4.200 KiB.
- En düşük host gecikmesi: TinyML-MLP/PTQ =
  0.018750 ms.
- En yüksek host throughput: TinyML-MLP/P75 =
  4276226.411 örnek/s.
- Global Pareto çözüm sayısı: 18.

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

- Doğruluk: Compact-DNN/QAT
- Minimum artifact: Tiny-1D-CNN/P75
- Düşük host gecikmesi: TinyML-MLP/PTQ
- Yüksek host throughput: TinyML-MLP/P75
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
En yüksek ortalama Macro F1 Compact-DNN/QAT ile
0.991846 olarak elde edilmiştir. En küçük artifact, en düşük
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
