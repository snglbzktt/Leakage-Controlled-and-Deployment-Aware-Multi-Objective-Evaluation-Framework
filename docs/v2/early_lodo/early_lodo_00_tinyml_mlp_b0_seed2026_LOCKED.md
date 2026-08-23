# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_00_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T19:26:35.666613+00:00`
- Held-out cihaz: `Danmini_Doorbell`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `11`
- Seçilen epoch: `7`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.997172320`
- Test fingerprint Macro-F1: `0.981657645`
- Validation-test farkı: `0.015514675`
- Test accuracy: `0.979988741`
- Record-weighted Macro-F1: `0.982494739`
- Benign FNR: `0.00017328877336304`
- Gafgyt FNR: `0.06543131915117549`
- Mirai FNR: `6.287379236313449e-05`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_00_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_00_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_00_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_00_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_00_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_00_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_00_tinyml_mlp_b0_seed2026\release_manifest.csv`
