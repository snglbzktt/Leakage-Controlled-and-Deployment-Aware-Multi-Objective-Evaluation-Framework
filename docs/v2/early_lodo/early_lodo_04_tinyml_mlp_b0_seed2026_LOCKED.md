# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_04_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:56:03.507762+00:00`
- Held-out cihaz: `Provision_PT_737E_Security_Camera`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `13`
- Seçilen epoch: `9`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.994909481`
- Test fingerprint Macro-F1: `0.962982630`
- Validation-test farkı: `0.031926851`
- Test accuracy: `0.968993707`
- Record-weighted Macro-F1: `0.965109223`
- Benign FNR: `0.0006887926190433033`
- Gafgyt FNR: `0.06293560540269123`
- Mirai FNR: `0.011653402444898053`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_04_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_04_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_04_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_04_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_04_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_04_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_04_tinyml_mlp_b0_seed2026\release_manifest.csv`
