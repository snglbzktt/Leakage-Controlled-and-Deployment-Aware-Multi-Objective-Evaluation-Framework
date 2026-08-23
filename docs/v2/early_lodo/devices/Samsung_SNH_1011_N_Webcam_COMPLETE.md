# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Samsung_SNH_1011_N_Webcam`
- Tamamlama zamanı: `2026-08-05T22:22:33.741915+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1486393`
- Doğrulama parmak izi: `318155`
- Test parmak izi: `359368`
- Test kayıt ağırlığı toplamı: `375222`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.993996951 | 0.986039380 | 0.01611315358817852 | None |
| 2 | tinyml_mlp_b0 | 0.993494249 | 0.988112464 | 0.013734268533193032 | None |
| 3 | hist_gradient_boosting_b0 | 0.992206735 | 0.977282340 | 0.02635171993066696 | None |

Cihaz içindeki en yüksek Macro-F1 değeri
`compact_dnn_b0` tarafından
`0.993996951` olarak üretildi.

Bu sıralama yalnız `Samsung_SNH_1011_N_Webcam` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Samsung_SNH_1011_N_Webcam\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
