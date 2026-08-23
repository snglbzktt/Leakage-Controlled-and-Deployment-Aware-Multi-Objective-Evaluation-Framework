# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_07_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:25:52.982161+00:00`
- Held-out cihaz: `SimpleHome_XCS7_1002_WHT_Security_Camera`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `10`
- Seçilen epoch: `6`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.998480762`
- Test fingerprint Macro-F1: `0.984102342`
- Validation-test farkı: `0.014378419`
- Test accuracy: `0.988288725`
- Record-weighted Macro-F1: `0.980651179`
- Benign FNR: `0.017716903515332835`
- Gafgyt FNR: `0.01753861484803189`
- Mirai FNR: `0.00790650913398591`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_07_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_07_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_07_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_07_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_07_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_07_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_07_tinyml_mlp_b0_seed2026\release_manifest.csv`
