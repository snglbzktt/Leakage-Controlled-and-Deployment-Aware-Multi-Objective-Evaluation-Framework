# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_01_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T20:07:39.774786+00:00`
- Held-out cihaz: `Ecobee_Thermostat`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `13`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.998548111`
- Test fingerprint Macro-F1: `0.988146765`
- Validation-test farkı: `0.010401346`
- Test accuracy: `0.991898038`
- Record-weighted Macro-F1: `0.988301487`
- Benign FNR: `0.0003813591640607124`
- Gafgyt FNR: `0.016465809527006613`
- Mirai FNR: `0.0034346546697830447`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_01_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_01_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_01_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_01_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_01_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_01_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_01_tinyml_mlp_b0_seed2026\release_manifest.csv`
