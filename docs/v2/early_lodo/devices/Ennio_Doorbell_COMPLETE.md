# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `Ennio_Doorbell`
- Tamamlama zamanı: `2026-08-05T21:31:12.283408+00:00`
- Seed: `2026`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `1500913`
- Doğrulama parmak izi: `321267`
- Test parmak izi: `338599`
- Test kayıt ağırlığı toplamı: `355500`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.993244568 | 0.985829846 | 0.015759146934750272 | None |
| 2 | tinyml_mlp_b0 | 0.992445435 | 0.982395105 | 0.01958514326047449 | None |
| 3 | hist_gradient_boosting_b0 | 0.903349652 | 0.713631759 | 0.3198137135582096 | None |

Cihaz içindeki en yüksek Macro-F1 değeri
`compact_dnn_b0` tarafından
`0.993244568` olarak üretildi.

Bu sıralama yalnız `Ennio_Doorbell` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `0`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `results\v2\early_lodo\devices\Ennio_Doorbell\cache_release_manifest.csv`

Cache, kilitli birincil verilerden tekrar üretilebilir.
