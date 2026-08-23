# Faz 2D — Erken LODO Protokolü Kilitlendi

Kilitleme zamanı: `2026-08-05T18:51:01.165582+00:00`

## Deney kapsamı

- Görev: `family_3`
- Cihaz sayısı: `9`
- Model sayısı: `3`
- Seed: `2026`
- Toplam çalışma: `27`

## Modeller

1. `tinyml_mlp_b0`
2. `compact_dnn_b0`
3. `hist_gradient_boosting_b0`

## Katlama politikası

Test kümesi, tutulan cihazda görülen bütün exact parmak
izlerini içerir. Bu parmak izlerinin tamamı eğitim ve doğrulama
kümelerinden çıkarılır. Bu nedenle eğitim–test exact fingerprint
örtüşmesi sıfırdır.

## Ana değerlendirme

Ana değerlendirme parmak izi seviyesindedir. İkincil değerlendirme,
tutulan cihazın occurrence sayıları kullanılarak kayıt ağırlıklı
olarak yapılır.

Ennio_Doorbell ve Samsung_SNH_1011_N_Webcam katlarında Mirai sınıfı
bulunmadığı için Mirai Recall, F1 ve FNR değerleri `N/A` raporlanır.

## Kırılganlık kuralları

- Device-macro Macro-F1 `< 0.85`
- Uygun saldırı hücrelerinin medyan FNR değeri `> 0.40`
- En az `6` uygun cihaz-saldırı
  hücresinde FNR `> 0.40`

Tek bir yüksek FNR hücresi yerel ciddi başarısızlık olarak raporlanır;
tek başına genel kırılganlık sonucuna dönüştürülmez.

## Dosyalar

- `results\v2\audit\early_lodo_protocol_locked_v2.json`
- `results\v2\audit\early_lodo_run_matrix_v2.csv`
- `results\v2\audit\early_lodo_protocol_release_manifest_v2.csv`
