# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Provision_PT_737E_Security_Camera`
- Tamamlama zamanı: `2026-08-05T22:02:07.081410+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1172731`
- Doğrulama parmak izi: `250895`
- Test parmak izi: `807836`
- Test kayıt ağırlığı toplamı: `828260`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | tinyml_mlp_b0 | 0.962982630 | 0.968993707 | 0.06293560540269123 | 0.011653402444898053 |
| 2 | compact_dnn_b0 | 0.962962680 | 0.960141662 | 0.09554186390952987 | 0.00433246943877434 |
| 3 | hist_gradient_boosting_b0 | 0.924677990 | 0.920704202 | 0.200668230924944 | 0.0011811655695970276 |

Cihaz içindeki en yüksek Macro-F1 değeri
`tinyml_mlp_b0` tarafından
`0.962982630` olarak üretildi.

Bu sıralama yalnız `Provision_PT_737E_Security_Camera` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Provision_PT_737E_Security_Camera\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
