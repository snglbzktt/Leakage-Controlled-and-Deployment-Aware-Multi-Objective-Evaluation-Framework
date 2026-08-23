# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `SimpleHome_XCS7_1003_WHT_Security_Camera`
- Tamamlama zamanı: `2026-08-05T22:41:28.150038+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1152607`
- Doğrulama parmak izi: `246508`
- Test parmak izi: `836326`
- Test kayıt ağırlığı toplamı: `850826`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.992487299 | 0.994646824 | 0.009214904622277864 | 0.0032416579264266015 |
| 2 | tinyml_mlp_b0 | 0.990981594 | 0.993332743 | 0.011774783382202747 | 0.003865128384415181 |
| 3 | hist_gradient_boosting_b0 | 0.974522158 | 0.982394425 | 0.048443316970315946 | 3.88455114011576e-05 |

Cihaz içindeki en yüksek Macro-F1 değeri
`compact_dnn_b0` tarafından
`0.992487299` olarak üretildi.

Bu sıralama yalnız `SimpleHome_XCS7_1003_WHT_Security_Camera` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\SimpleHome_XCS7_1003_WHT_Security_Camera\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
