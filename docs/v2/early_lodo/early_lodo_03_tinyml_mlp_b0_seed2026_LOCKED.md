# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_03_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:36:11.213714+00:00`
- Held-out cihaz: `Philips_B120N10_Baby_Monitor`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `13`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.998433410`
- Test fingerprint Macro-F1: `0.990533447`
- Validation-test farkı: `0.007899963`
- Test accuracy: `0.990697402`
- Record-weighted Macro-F1: `0.990425179`
- Benign FNR: `0.008859096166306368`
- Gafgyt FNR: `0.02835234486688594`
- Mirai FNR: `6.22222513320474e-05`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_03_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_03_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_03_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_03_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_03_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_03_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_03_tinyml_mlp_b0_seed2026\release_manifest.csv`
