# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Philips_B120N10_Baby_Monitor`
- Tamamlama zamanı: `2026-08-05T21:46:30.511617+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `984133`
- Doğrulama parmak izi: `210978`
- Test parmak izi: `1075936`
- Test kayıt ağırlığı toplamı: `1098677`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.993263010 | 0.993567461 | 0.018664960460675617 | 6.22222513320474e-05 |
| 2 | tinyml_mlp_b0 | 0.990533447 | 0.990697402 | 0.02835234486688594 | 6.22222513320474e-05 |
| 3 | hist_gradient_boosting_b0 | 0.975964181 | 0.973872052 | 0.09364138349312352 | 0.0 |

Cihaz içindeki en yüksek Macro-F1 değeri
`compact_dnn_b0` tarafından
`0.993263010` olarak üretildi.

Bu sıralama yalnız `Philips_B120N10_Baby_Monitor` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Philips_B120N10_Baby_Monitor\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
