# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Provision_PT_838_Security_Camera`
- Tamamlama zamanı: `2026-08-05T22:12:43.628775+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1164353`
- Doğrulama parmak izi: `249102`
- Test parmak izi: `820042`
- Test kayıt ağırlığı toplamı: `836891`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.993238069 | 0.992367464 | 0.020400390954130296 | 0.0003517050708417863 |
| 2 | tinyml_mlp_b0 | 0.992204182 | 0.991250448 | 0.023029220450945367 | 0.0006312057893915502 |
| 3 | hist_gradient_boosting_b0 | 0.962810801 | 0.953759685 | 0.12779818678170604 | 0.0 |

Cihaz içindeki en yüksek Macro-F1 değeri
`compact_dnn_b0` tarafından
`0.993238069` olarak üretildi.

Bu sıralama yalnız `Provision_PT_838_Security_Camera` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Provision_PT_838_Security_Camera\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
