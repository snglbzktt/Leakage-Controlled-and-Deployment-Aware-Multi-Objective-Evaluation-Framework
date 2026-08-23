# 33. Aşama — Yayın İçin Sonuç Özeti

## 1. Sınıflandırma performansı

Toplam 3 model, 10 varyant ve 5 eşleştirilmiş seed üzerinden 150 test sonucu analiz edilmiştir. Birincil performans ölçütü Macro F1'dir.

### TinyML-MLP

İlk üç varyant: 1. QAT (0.989831 ± 0.002818); 2. P25-QAT (0.989219 ± 0.002738); 3. P25 (0.989068 ± 0.003109).

Friedman testi varyantlar arasında anlamlı fark göstermiştir: χ²(9) = 41.552727, p = 0.00000397. Kendall W = 0.923394 (large) olarak bulunmuştur.

Friedman kapı kontrolü ve Holm düzeltmesi sonrasında 45 ikili karşılaştırmanın 0 tanesi anlamlı kalmıştır.

### Compact-DNN

İlk üç varyant: 1. QAT (0.991846 ± 0.002950); 2. P25-QAT (0.991292 ± 0.003156); 3. P25 (0.991031 ± 0.002813).

Friedman testi varyantlar arasında anlamlı fark göstermiştir: χ²(9) = 37.538182, p = 0.00002109. Kendall W = 0.834182 (large) olarak bulunmuştur.

Friedman kapı kontrolü ve Holm düzeltmesi sonrasında 45 ikili karşılaştırmanın 0 tanesi anlamlı kalmıştır.

### Tiny-1D-CNN

İlk üç varyant: 1. QAT (0.932389 ± 0.033147); 2. B0 (0.917594 ± 0.023553); 3. P25-QAT (0.900939 ± 0.086503).

Friedman testi varyantlar arasında anlamlı fark göstermiştir: χ²(9) = 40.063107, p = 0.00000740. Kendall W = 0.890291 (large) olarak bulunmuştur.

Friedman kapı kontrolü ve Holm düzeltmesi sonrasında 45 ikili karşılaştırmanın 0 tanesi anlamlı kalmıştır.

## 2. Mühendislik ödünleşimleri

En yüksek ortalama Macro F1, Compact-DNN/QAT tarafından 0.991846 değeriyle elde edilmiştir.

En küçük deployment artifact, Tiny-1D-CNN/P75 için 4.200 KiB olarak ölçülmüştür.

En düşük host CPU medyan gecikmesi, TinyML-MLP/PTQ için 0.018750 ms'dir.

En yüksek host CPU throughput değeri, TinyML-MLP/P75 için 4276226.411 örnek/s olarak bulunmuştur.

Bu sonuçlar, en yüksek doğruluk, en küçük model boyutu, en düşük gecikme ve en yüksek throughput hedeflerinin aynı model-varyant kombinasyonunda birleşmediğini göstermektedir.

## 3. Global Pareto-optimal çözümler

- TinyML-MLP/P25: Macro F1=0.989068, artifact=29.396 KiB, medyan gecikme=0.047840 ms, MAC=6744.
- TinyML-MLP/P50: Macro F1=0.988128, artifact=19.458 KiB, medyan gecikme=0.052060 ms, MAC=4240.
- TinyML-MLP/P75: Macro F1=0.983060, artifact=10.646 KiB, medyan gecikme=0.050440 ms, MAC=1992.
- TinyML-MLP/DQ: Macro F1=0.977597, artifact=14.531 KiB, medyan gecikme=0.023200 ms, MAC=9504.
- TinyML-MLP/PTQ: Macro F1=0.970950, artifact=15.548 KiB, medyan gecikme=0.018750 ms, MAC=9504.
- TinyML-MLP/QAT: Macro F1=0.989831, artifact=17.726 KiB, medyan gecikme=0.142980 ms, MAC=9504.
- TinyML-MLP/P25-QAT: Macro F1=0.989219, artifact=14.601 KiB, medyan gecikme=0.144920 ms, MAC=6744.
- TinyML-MLP/P50-QAT: Macro F1=0.988527, artifact=11.663 KiB, medyan gecikme=0.143580 ms, MAC=4240.
- TinyML-MLP/P75-QAT: Macro F1=0.982931, artifact=8.976 KiB, medyan gecikme=0.144780 ms, MAC=1992.
- Compact-DNN/P25: Macro F1=0.991031, artifact=69.825 KiB, medyan gecikme=0.064070 ms, MAC=16872.
- Compact-DNN/DQ: Macro F1=0.980241, artifact=32.479 KiB, medyan gecikme=0.027320 ms, MAC=25056.
- Compact-DNN/PTQ: Macro F1=0.979748, artifact=34.413 KiB, medyan gecikme=0.025060 ms, MAC=25056.
- Compact-DNN/QAT: Macro F1=0.991846, artifact=37.083 KiB, medyan gecikme=0.192740 ms, MAC=25056.
- Compact-DNN/P25-QAT: Macro F1=0.991292, artifact=28.021 KiB, medyan gecikme=0.202670 ms, MAC=16872.
- Tiny-1D-CNN/B0: Macro F1=0.917594, artifact=10.138 KiB, medyan gecikme=0.113230 ms, MAC=58816.
- Tiny-1D-CNN/P25: Macro F1=0.887149, artifact=7.450 KiB, medyan gecikme=0.134730 ms, MAC=33960.
- Tiny-1D-CNN/P50: Macro F1=0.615113, artifact=5.388 KiB, medyan gecikme=0.107280 ms, MAC=15872.
- Tiny-1D-CNN/P75: Macro F1=0.488134, artifact=4.200 KiB, medyan gecikme=0.111300 ms, MAC=4552.

## 4. Geçerlilik sınırları

Gecikme ve throughput değerleri yalnızca çalışmanın yürütüldüğü host CPU, işletim sistemi, PyTorch ve ONNX Runtime ortamı için geçerlidir. Bunlar MCU gecikmesi veya enerji ölçümü değildir. Beş seed kullanılması nedeniyle özellikle ikili Wilcoxon karşılaştırmalarında istatistiksel test gücü sınırlıdır; bu nedenle p-değerleri eşleştirilmiş farklar, güven aralıkları ve etki büyüklükleriyle birlikte yorumlanmalıdır.

## 5. Üretilen şekiller

- Şekil 1: Macro F1–artifact boyutu ödünleşimi
- Şekil 2: Macro F1–host CPU gecikme ödünleşimi
- Şekil 3: Model–varyant Macro F1 ısı haritası
