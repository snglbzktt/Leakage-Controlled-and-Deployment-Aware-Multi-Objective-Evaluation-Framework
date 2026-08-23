# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `SimpleHome_XCS7_1002_WHT_Security_Camera`
- Tamamlama zamanı: `2026-08-05T22:31:58.742052+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1145094`
- Doğrulama parmak izi: `244988`
- Test parmak izi: `847047`
- Test kayıt ağırlığı toplamı: `863056`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | hist_gradient_boosting_b0 | 0.995889158 | 0.995507923 | 0.012800027489991924 | 0.00013638630837333997 |
| 2 | compact_dnn_b0 | 0.987226238 | 0.990109167 | 0.016885727539817534 | 0.0052664598790448285 |
| 3 | tinyml_mlp_b0 | 0.984102342 | 0.988288725 | 0.01753861484803189 | 0.00790650913398591 |

Cihaz içindeki en yüksek Macro-F1 değeri
`hist_gradient_boosting_b0` tarafından
`0.995889158` olarak üretildi.

Bu sıralama yalnız `SimpleHome_XCS7_1002_WHT_Security_Camera` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\SimpleHome_XCS7_1002_WHT_Security_Camera\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
