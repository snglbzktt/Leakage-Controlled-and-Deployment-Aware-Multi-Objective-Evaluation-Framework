# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Ecobee_Thermostat`
- Tamamlama zamanı: `2026-08-05T21:14:04.122207+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1162046`
- Doğrulama parmak izi: `248309`
- Test parmak izi: `823134`
- Test kayıt ağırlığı toplamı: `835876`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | hist_gradient_boosting_b0 | 0.992276939 | 0.992662191 | 0.020175232468360803 | 5.857853331068297e-05 |
| 2 | compact_dnn_b0 | 0.990265033 | 0.993423914 | 0.01446842794320051 | 0.002140069083616951 |
| 3 | tinyml_mlp_b0 | 0.988146765 | 0.991898038 | 0.016465809527006613 | 0.0034346546697830447 |

Cihaz içindeki en yüksek Macro-F1 değeri
`hist_gradient_boosting_b0` tarafından
`0.992276939` olarak üretildi.

Bu sıralama yalnız `Ecobee_Thermostat` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Ecobee_Thermostat\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
