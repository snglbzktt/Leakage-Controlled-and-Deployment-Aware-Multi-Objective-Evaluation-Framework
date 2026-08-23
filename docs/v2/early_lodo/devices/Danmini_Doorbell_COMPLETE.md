# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Danmini_Doorbell`
- Tamamlama zamanı: `2026-08-05T19:53:47.589653+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1039932`
- Doğrulama parmak izi: `222798`
- Test parmak izi: `996539`
- Test kayıt ağırlığı toplamı: `1018298`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | hist_gradient_boosting_b0 | 0.984940617 | 0.982505451 | 0.05734038494428438 | 0.0 |
| 2 | tinyml_mlp_b0 | 0.981657645 | 0.979988741 | 0.06543131915117549 | 6.287379236313449e-05 |
| 3 | compact_dnn_b0 | 0.975706276 | 0.973972920 | 0.08512912604754574 | 5.213924244747738e-05 |

Cihaz içindeki en yüksek Macro-F1 değeri
`hist_gradient_boosting_b0` tarafından
`0.984940617` olarak üretildi.

Bu sıralama yalnız `Danmini_Doorbell` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Danmini_Doorbell\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
