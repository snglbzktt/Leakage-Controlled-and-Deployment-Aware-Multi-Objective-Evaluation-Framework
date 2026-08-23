# Faz 2F — Erken LODO Smoke Testi Tamamlandı

Kilitleme zamanı: `2026-08-05T19:11:20.307441+00:00`

## Smoke testi

- Tutulan cihaz: `Danmini_Doorbell`
- Model: `tinyml_mlp`
- Seed: `2026`
- Epoch: `1`
- Eğitim örneği: `12288`
- Doğrulama örneği: `6144`
- Test örneği: `6144`
- Model parametresi: `9603`

## Örtüşme denetimi

- Eğitim–test örtüşmesi: `0`
- Doğrulama–test örtüşmesi: `0`
- Eğitim–doğrulama örtüşmesi: `0`

## Sonuç

Veri okuma, cihaz üyeliği, strict LODO filtreleme, yalnız eğitim
verisiyle scaler öğrenme, model oluşturma, optimizasyon ve metrik
hesaplama zinciri başarıyla doğrulandı.

Smoke testinde üretilen Macro-F1 ve diğer performans değerleri bilimsel
deney sonucu değildir ve makalede performans sonucu olarak kullanılamaz.

## Dosyalar

- `results\v2\smoke\early_lodo\danmini_tinyml_mlp_smoke_v2.json`
- `results\v2\smoke\early_lodo\danmini_tinyml_mlp_smoke_v2.txt`
- `results\v2\audit\early_lodo_smoke_test_locked_summary_v2.json`
- `results\v2\audit\early_lodo_smoke_test_release_manifest_v2.csv`
