# Bilimsel Katkı Beyanı

## Çalışmanın Geçici Başlığı

**Kısıtlı IoT Cihazları İçin TinyML Tabanlı Saldırı Tespit Sistemi: Kuantize Edilmiş ve Budanmış Yapay Sinir Ağı Mimarilerinin Simülasyon Tabanlı Karşılaştırmalı İncelemesi**

### İngilizce Başlık

**Simulation-Based Comparative Analysis of Quantized and Pruned TinyML Neural Architectures for Intrusion Detection in Resource-Constrained IoT Devices**

---

## 1. Araştırmanın Kapsamı

Bu çalışma, kısıtlı IoT cihazlarına yönelik TinyML tabanlı saldırı tespit modellerini yazılım tabanlı bir deney ve simülasyon ortamında incelemeyi amaçlamaktadır.

Çalışma kapsamında modeller fiziksel bir mikrodenetleyici üzerinde çalıştırılmayacaktır. Bunun yerine modellerin kısıtlı cihazlarda kullanılabilirliği aşağıdaki ölçülebilir ve tahmin edilebilir kaynak göstergeleri üzerinden değerlendirilecektir:

- Toplam parametre sayısı
- Eğitilebilir parametre sayısı
- Sıfır olmayan parametre sayısı
- Model dosya boyutu
- Ağırlıkların bit genişliği
- MAC sayısı
- FLOP sayısı
- Bellek gereksinimi tahmini
- Kontrollü yazılım ortamında çıkarım gecikmesi
- Kontrollü yazılım ortamında işlem hacmi
- Sıkıştırma oranı
- Hedef cihaz RAM ve Flash sınırlarına teorik uygunluk

Çalışmanın sonuçları gerçek cihaz ölçümleri olarak değil, simülasyon ve yazılım tabanlı kaynak değerlendirme sonuçları olarak raporlanacaktır.

---

## 2. Araştırma Problemi

Nesnelerin İnterneti cihazları genellikle sınırlı işlem gücü, RAM, Flash bellek ve enerji bütçesiyle çalışmaktadır. Buna karşın bu cihazlar botnet, hizmet engelleme, veri sızdırma, kötü amaçlı trafik üretimi ve yetkisiz erişim gibi çeşitli siber saldırılara maruz kalabilmektedir.

Derin öğrenme tabanlı saldırı tespit sistemleri yüksek sınıflandırma başarısı sağlayabilse de bu modellerin hesaplama ve bellek gereksinimleri, kısıtlı IoT cihazlarında doğrudan kullanılmalarını zorlaştırmaktadır.

Budama ve kuantizasyon gibi model sıkıştırma yöntemleri, modellerin kaynak gereksinimini azaltabilir. Ancak bu yöntemler saldırı tespit başarısında, özellikle güvenlik açısından kritik saldırı sınıflarında performans kaybına neden olabilir.

Bu nedenle yalnızca genel doğruluğun incelenmesi yeterli değildir. Model sıkıştırmanın saldırı sınıfı bazındaki etkisi, yanlış negatif oranı, model büyüklüğü, hesaplama karmaşıklığı ve çıkarım maliyetiyle birlikte değerlendirilmelidir.

---

## 3. Temel Araştırma Amacı

Bu çalışmanın temel amacı, yazılım tabanlı kontrollü bir simülasyon ortamında farklı hafif sinir ağı mimarilerinin, budama oranlarının ve kuantizasyon yöntemlerinin saldırı tespit performansı ile kaynak verimliliği üzerindeki etkisini karşılaştırmaktır.

Çalışma sonucunda aşağıdaki özellikler arasında en uygun dengeyi sağlayan model yapılandırmalarının belirlenmesi hedeflenmektedir:

- Yüksek saldırı tespit başarısı
- Düşük yanlış negatif oranı
- Düşük yanlış pozitif oranı
- Küçük model boyutu
- Düşük parametre sayısı
- Düşük işlem karmaşıklığı
- Düşük tahmini bellek gereksinimi
- Düşük yazılım tabanlı çıkarım gecikmesi

---

## 4. Temel Araştırma Sorusu

Bu çalışma aşağıdaki temel soruya cevap arayacaktır:

> Kontrollü bir yazılım ve simülasyon ortamında hangi hafif sinir ağı mimarisi, budama oranı ve kuantizasyon yöntemi; saldırı tespit başarısı, model boyutu, işlem karmaşıklığı ve tahmini kaynak gereksinimi arasında en uygun dengeyi sağlamaktadır?

---

## 5. Önerilen Bilimsel Katkılar

### K1. Hafif Sinir Ağı Mimarilerinin Kontrollü Karşılaştırılması

Birden fazla hafif sinir ağı mimarisi aynı veri hazırlama, eğitim, doğrulama ve test protokolü altında karşılaştırılacaktır.

Değerlendirilmesi planlanan mimariler:

- Hafif çok katmanlı algılayıcı
- Kompakt derin sinir ağı
- Bir boyutlu evrişimsel sinir ağı
- Uygun olması durumunda kompakt residual ağ

Bütün mimariler aynı veri bölmeleri, random seed değerleri, optimizasyon koşulları ve değerlendirme metrikleri kullanılarak karşılaştırılacaktır.

---

### K2. Budama Oranlarının Sistematik Analizi

Sinir ağı ağırlıklarına farklı oranlarda budama uygulanacaktır.

Planlanan temel budama oranları:

- %25
- %50
- %75

Gerekli görülmesi durumunda ara oranlar da değerlendirilecektir.

Budamanın aşağıdaki unsurlar üzerindeki etkisi incelenecektir:

- Macro F1-score
- Sınıf bazlı recall
- Yanlış negatif oranı
- Sıfır olmayan parametre sayısı
- Model boyutu
- MAC ve FLOP değerleri
- Yazılım tabanlı çıkarım gecikmesi

Budama sonrası fine-tuning işleminin sağladığı performans geri kazanımı ayrıca analiz edilecektir.

---

### K3. Kuantizasyon Yöntemlerinin Karşılaştırılması

Farklı kuantizasyon yöntemleri ortak bir deney protokolü altında karşılaştırılacaktır.

Değerlendirilmesi planlanan model biçimleri:

- FP32 temel model
- Dinamik kuantizasyon
- Post-training quantization
- INT8 kuantizasyon
- Quantization-aware training

Kuantizasyonun aşağıdaki değişkenler üzerindeki etkisi incelenecektir:

- Sınıflandırma performansı
- Model dosya boyutu
- Ağırlık bellek gereksinimi
- Çıkarım gecikmesi
- İşlem hacmi
- Saldırı sınıfı bazındaki performans kaybı

---

### K4. Budama ve Kuantizasyonun Birleşik Etkisinin İncelenmesi

Budama ve kuantizasyon yöntemleri ayrı ayrı ve birlikte uygulanacaktır.

Temel karşılaştırma grupları:

1. FP32 temel model
2. Yalnızca budanmış model
3. Yalnızca kuantize edilmiş model
4. Budanmış ve kuantize edilmiş model

Birleşik model sıkıştırmanın, tek başına uygulanan yöntemlere göre sağladığı avantaj ve kayıplar incelenecektir.

Budama ve kuantizasyon işlemlerinin uygulanma sırası da uygun deney koşullarında değerlendirilecektir.

---

### K5. Güvenlik Performansının Sınıf Bazında Analizi

Genel doğruluk değeri, kritik saldırı sınıflarındaki performans kaybını gizleyebilir.

Bu nedenle her model için aşağıdaki sınıf bazlı metrikler raporlanacaktır:

- Precision
- Recall
- F1-score
- False Negative Rate
- False Positive Rate
- Confusion Matrix

Budama ve kuantizasyonun hangi saldırı türlerini daha fazla etkilediği ayrıca analiz edilecektir.

Bu değerlendirme, küçük bir genel performans kaybının güvenlik açısından kritik bir saldırı sınıfında büyük bir algılama kaybına karşılık gelip gelmediğini gösterecektir.

---

### K6. Doğruluk–Verimlilik Pareto Analizi

Modeller yalnızca doğruluk veya F1-score değerine göre sıralanmayacaktır.

Aşağıdaki hedefler birlikte değerlendirilecektir:

- Yüksek Macro F1-score
- Yüksek saldırı algılama oranı
- Düşük yanlış negatif oranı
- Küçük model boyutu
- Düşük parametre sayısı
- Düşük MAC/FLOP değeri
- Düşük tahmini bellek ihtiyacı
- Düşük yazılım tabanlı çıkarım gecikmesi

Bu değişkenler kullanılarak Pareto-optimal model yapılandırmaları belirlenecektir.

Böylece tek bir “en iyi model” iddiası yerine farklı kaynak sınırları için uygun model seçenekleri sunulacaktır.

---

### K7. Hedef Cihaz Profillerine Göre Teorik Uygunluk Analizi

Gerçek mikrodenetleyici kullanılmadan, farklı kısıtlı cihaz profilleri tanımlanacaktır.

Örnek profiller:

- Çok düşük kaynaklı cihaz profili
- Düşük kaynaklı cihaz profili
- Orta düzey kaynaklı edge cihaz profili

Her profil için aşağıdaki sınırlar tanımlanabilir:

- Maksimum Flash kapasitesi
- Maksimum RAM kapasitesi
- Maksimum model boyutu
- Maksimum tahmini işlem bütçesi

Modellerin bu kaynak profillerine teorik olarak uyup uymadığı analiz edilecektir.

Bu sonuçlar gerçek cihaz performansı olarak değil, kaynak bütçesine dayalı dağıtım uygunluğu analizi olarak sunulacaktır.

---

### K8. İstatistiksel Olarak Güvenilir Deney Tasarımı

Her temel model deneyi birden fazla random seed ile tekrarlanacaktır.

Temel deneylerin en az beş farklı seed ile yürütülmesi hedeflenmektedir.

Sonuçlar aşağıdaki biçimde raporlanacaktır:

- Ortalama
- Standart sapma
- Medyan
- %95 güven aralığı
- Etki büyüklüğü

Model karşılaştırmaları için deney yapısına uygun olarak aşağıdaki yöntemler kullanılabilir:

- Paired t-test
- Wilcoxon signed-rank testi
- Friedman testi
- Post-hoc karşılaştırmalar
- Çoklu karşılaştırma düzeltmeleri

İstatistiksel değerlendirme yalnızca p-value üzerinden yapılmayacaktır. Etki büyüklüğü ve pratik performans farkı da dikkate alınacaktır.

---

### K9. Ablation Study

Çalışmanın hangi bileşenlerinin sonuca ne kadar katkı sağladığını göstermek amacıyla ablation study gerçekleştirilecektir.

Aşağıdaki durumlar karşılaştırılacaktır:

- Budama olmadan
- Kuantizasyon olmadan
- Fine-tuning olmadan
- Yalnızca budama
- Yalnızca kuantizasyon
- Budama ve kuantizasyon birlikte
- Farklı budama oranları
- Farklı mimari derinlikleri
- Farklı özellik alt kümeleri
- Farklı bit genişlikleri

Bu analiz, performans ve verimlilik kazanımlarının hangi bileşenlerden kaynaklandığını gösterecektir.

---

### K10. Açıklanabilirlik ve Özellik Önem Analizi

Modellerin saldırı kararlarını hangi ağ trafiği özelliklerine dayanarak verdiği analiz edilecektir.

Uygun yöntemler:

- SHAP
- Permutation Feature Importance
- Gradient tabanlı özellik önem yöntemleri

FP32, budanmış ve kuantize edilmiş modellerin özellik önem dağılımları karşılaştırılacaktır.

Bu sayede model sıkıştırma sonrasında karar mekanizmasının değişip değişmediği incelenecektir.

---

### K11. Tekrarlanabilir Simülasyon ve Deney Altyapısı

Çalışmada aşağıdaki işlemleri destekleyen modüler bir araştırma altyapısı geliştirilecektir:

- Veri seti yükleme
- Veri doğrulama
- Veri sızıntısı kontrolü
- Tekrarlanabilir veri bölme
- Özellik ölçekleme
- Model eğitimi
- Model değerlendirmesi
- Budama
- Kuantizasyon
- Model dönüştürme
- Parametre ve karmaşıklık analizi
- Bellek gereksinimi tahmini
- Yazılım tabanlı gecikme ölçümü
- Çoklu random seed deneyleri
- Deney takibi
- Model kayıt sistemi
- CSV ve JSON sonuç üretimi
- Grafik ve tablo üretimi
- İstatistiksel analiz

Tüm deney konfigürasyonları ve random seed değerleri kaydedilecektir.

---

## 6. Kullanılacak Değerlendirme Ölçütleri

### 6.1. Sınıflandırma Metrikleri

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
- Sınıf bazlı recall
- Confusion Matrix

Dengesiz veri kümelerinde Accuracy tek başına temel başarı ölçütü olarak kullanılmayacaktır.

---

### 6.2. Model Karmaşıklığı Metrikleri

- Toplam parametre sayısı
- Eğitilebilir parametre sayısı
- Sıfır olmayan parametre sayısı
- Sıfırlık oranı
- MAC sayısı
- FLOP sayısı
- Model dosya boyutu
- Sıkıştırma oranı
- Ağırlıkların tahmini bellek gereksinimi

---

### 6.3. Simülasyon Tabanlı Çıkarım Metrikleri

- Ortalama çıkarım gecikmesi
- Medyan çıkarım gecikmesi
- Standart sapma
- p95 çıkarım gecikmesi
- Saniyedeki çıkarım sayısı
- Kontrollü CPU ortamındaki throughput
- Pik bellek kullanımı

Bu değerler yalnızca ölçüm yapılan yazılım ve bilgisayar ortamı için geçerli olacaktır.

---

## 7. Simülasyon Ortamının Kontrolü

Yazılım tabanlı gecikme ölçümlerinin karşılaştırılabilir olması için aşağıdaki koşullar sabit tutulacaktır:

- Aynı bilgisayar
- Aynı işlemci
- Aynı Python sürümü
- Aynı PyTorch sürümü
- Aynı ONNX Runtime sürümü
- Aynı işletim sistemi
- Aynı thread sayısı
- Batch size = 1
- Aynı giriş boyutu
- Aynı warm-up sayısı
- Aynı ölçüm tekrar sayısı
- Arka plan uygulamalarının mümkün olduğunca sınırlandırılması

Her model için ilk çalıştırmalar ölçüme dahil edilmeden önce warm-up işlemi yapılacaktır.

Gecikme sonuçları birden fazla tekrar üzerinden raporlanacaktır.

---

## 8. Bilimsel Konumlandırma

Çalışmanın bilimsel katkısı yalnızca TinyML, budama, kuantizasyon veya ONNX kullanılmış olmasına dayandırılmayacaktır.

Bilimsel katkı aşağıdaki unsurlarla desteklenecektir:

1. Güncel literatürden çıkarılan açık bir araştırma boşluğu
2. Kontrollü ve adil model karşılaştırması
3. Veri sızıntısını önleyen deney protokolü
4. Çoklu random seed deneyleri
5. İstatistiksel güvenilirlik analizi
6. Sınıf bazlı güvenlik değerlendirmesi
7. Ablation study
8. Doğruluk–kaynak dengesi analizi
9. Hedef cihaz kaynak profillerine teorik uygunluk
10. Tekrarlanabilir kod ve deney konfigürasyonları
11. Açık şekilde belirtilmiş sınırlılıklar

---

## 9. Yapılmayacak İddialar

Gerçek donanım kullanılmadığı için aşağıdaki iddialarda bulunulmayacaktır:

- Model gerçek mikrodenetleyici üzerinde doğrulanmıştır.
- Sistem gerçek zamanlı olarak çalışmaktadır.
- Modelin gerçek enerji tüketimi ölçülmüştür.
- Model belirtilen cihazda kesin olarak belirli bir gecikmeyle çalışacaktır.
- Model gerçek IoT ortamında deployment-ready durumdadır.
- Masaüstü gecikme sonuçları mikrodenetleyici gecikmesini temsil etmektedir.
- Simülasyon sonuçları gerçek donanım sonuçlarıyla aynıdır.
- Model tüm IoT cihazlarına doğrudan uygulanabilir.
- Model enerji açısından en verimli çözümdür.

---

## 10. Kullanılabilecek Akademik İfadeler

Sonuçlar desteklediği takdirde aşağıdaki türde ifadeler kullanılabilir:

- Simülasyon tabanlı deneylerde daha düşük kaynak gereksinimi göstermiştir.
- Kontrollü yazılım ortamında daha düşük çıkarım gecikmesi elde edilmiştir.
- Model boyutunda belirgin bir azalma sağlanmıştır.
- Hedef cihaz kaynak profilleriyle teorik uyumluluk göstermiştir.
- Budama ve kuantizasyon arasında doğruluk–verimlilik dengesi belirlenmiştir.
- Sıkıştırma, belirli saldırı sınıflarında daha yüksek performans kaybına neden olmuştur.
- Sonuçlar gerçek cihaz doğrulaması için umut verici bir aday ortaya koymaktadır.

Bu ifadeler, gerçek donanım performansı iddiası taşımayacaktır.

---

## 11. Çalışmanın Sınırlılıkları

Çalışmanın temel sınırlılığı, modellerin fiziksel bir mikrodenetleyici veya gerçek kısıtlı IoT cihazı üzerinde doğrulanmamasıdır.

Bu nedenle:

- Gerçek Flash kullanımı doğrulanamayabilir.
- Gerçek RAM tüketimi yazılım tahmininden farklı olabilir.
- Masaüstü gecikmesi mikrodenetleyici gecikmesini temsil etmez.
- Gerçek enerji tüketimi belirlenemez.
- Derleyici ve donanım hızlandırıcı etkileri ölçülemez.
- İşletim sistemi ve gerçek ağ yükünün etkileri incelenemez.
- Simülasyon sonuçları gerçek dünya dağıtımını tam olarak temsil etmeyebilir.

Bu sınırlılıklar makalenin “Limitations” ve “Threats to Validity” bölümlerinde açıkça belirtilecektir.

---

## 12. Gelecek Çalışmalar

Gelecek çalışmalarda aşağıdaki doğrulamaların yapılması önerilecektir:

- Modelin gerçek mikrodenetleyici üzerinde çalıştırılması
- Gerçek RAM ve Flash kullanımının ölçülmesi
- Gerçek çıkarım gecikmesinin belirlenmesi
- Güç tüketimi ve çıkarım başına enerji ölçümü
- Farklı mikrodenetleyici ailelerinin karşılaştırılması
- Gerçek ağ trafiği üzerinde çevrim içi saldırı tespiti
- Uzun süreli kararlılık ve güvenilirlik testleri

Bu maddeler mevcut çalışmanın tamamlanmış deneyleri olarak değil, gelecek araştırma yönleri olarak sunulacaktır.

---

## 13. Yayına Hazır Olma Koşulları

Çalışma aşağıdaki koşullar sağlandığında simülasyon tabanlı akademik gönderime hazır kabul edilecektir:

- Araştırma boşluğu güncel literatürle doğrulanmış olmalıdır.
- Araştırma soruları açık ve test edilebilir olmalıdır.
- Veri işleme protokolü tekrarlanabilir olmalıdır.
- Veri sızıntısı kontrolleri tamamlanmış olmalıdır.
- Birden fazla hafif mimari karşılaştırılmış olmalıdır.
- FP32 temel modeller değerlendirilmiş olmalıdır.
- Budama deneyleri tamamlanmış olmalıdır.
- Kuantizasyon deneyleri tamamlanmış olmalıdır.
- Birleşik budama ve kuantizasyon deneyleri yapılmış olmalıdır.
- Deneyler birden fazla random seed ile tekrarlanmış olmalıdır.
- İstatistiksel analiz tamamlanmış olmalıdır.
- Ablation study yapılmış olmalıdır.
- Sınıf bazlı güvenlik analizi gerçekleştirilmiş olmalıdır.
- Parametre, model boyutu, MAC ve FLOP analizleri tamamlanmış olmalıdır.
- Kontrollü yazılım gecikme ölçümleri tamamlanmış olmalıdır.
- Hedef cihaz kaynak profillerine teorik uygunluk analiz edilmiş olmalıdır.
- Sonuçlar güncel çalışmalarla dikkatli biçimde karşılaştırılmış olmalıdır.
- Simülasyon sonuçları gerçek donanım sonucu gibi sunulmamalıdır.
- Çalışmanın sınırlılıkları açık şekilde belirtilmiş olmalıdır.

---

## 14. Başarı Ölçütü

Proje, aşağıdaki soruya bilimsel, ölçülebilir ve tekrarlanabilir bir cevap verdiğinde başarılı kabul edilecektir:

> Kontrollü bir simülasyon ortamında hangi hafif sinir ağı mimarisi, budama oranı ve kuantizasyon yöntemi; saldırı tespit performansı, model boyutu, işlem karmaşıklığı ve tahmini kaynak gereksinimi arasında en uygun dengeyi sağlamaktadır?