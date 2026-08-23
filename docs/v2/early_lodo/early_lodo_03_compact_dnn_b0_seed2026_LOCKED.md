# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_03_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:41:38.760365+00:00`
- Held-out cihaz: `Philips_B120N10_Baby_Monitor`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `15`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.998188250`
- Test fingerprint Macro-F1: `0.993263010`
- Validation-test farkı: `0.004925241`
- Test accuracy: `0.993567461`
- Record-weighted Macro-F1: `0.993221602`
- Benign FNR: `0.007763063079429094`
- Gafgyt FNR: `0.018664960460675617`
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

- `results\v2\early_lodo\runs\early_lodo_03_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_03_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_03_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_03_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_03_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_03_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_03_compact_dnn_b0_seed2026\release_manifest.csv`
