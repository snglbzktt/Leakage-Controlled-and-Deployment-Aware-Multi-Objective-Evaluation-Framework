# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_05_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:06:35.035798+00:00`
- Held-out cihaz: `Provision_PT_838_Security_Camera`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `13`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.997867491`
- Test fingerprint Macro-F1: `0.992204182`
- Validation-test farkı: `0.005663309`
- Test accuracy: `0.991250448`
- Record-weighted Macro-F1: `0.992366735`
- Benign FNR: `0.0007553593276238097`
- Gafgyt FNR: `0.023029220450945367`
- Mirai FNR: `0.0006312057893915502`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_05_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_05_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_05_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_05_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_05_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_05_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_05_tinyml_mlp_b0_seed2026\release_manifest.csv`
