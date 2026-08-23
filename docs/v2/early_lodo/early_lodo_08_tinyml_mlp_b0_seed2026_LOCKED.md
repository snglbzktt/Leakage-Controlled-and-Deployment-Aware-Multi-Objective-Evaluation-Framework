# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_08_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:37:03.777282+00:00`
- Held-out cihaz: `SimpleHome_XCS7_1003_WHT_Security_Camera`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `11`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.998406788`
- Test fingerprint Macro-F1: `0.990981594`
- Validation-test farkı: `0.007425194`
- Test accuracy: `0.993332743`
- Record-weighted Macro-F1: `0.991317857`
- Benign FNR: `0.0006690454950936663`
- Gafgyt FNR: `0.011774783382202747`
- Mirai FNR: `0.003865128384415181`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_08_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_08_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_08_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_08_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_08_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_08_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_08_tinyml_mlp_b0_seed2026\release_manifest.csv`
