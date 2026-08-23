# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_00_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T19:33:03.430137+00:00`
- Held-out cihaz: `Danmini_Doorbell`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `11`
- Seçilen epoch: `7`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.996835375`
- Test fingerprint Macro-F1: `0.975706276`
- Validation-test farkı: `0.021129099`
- Test accuracy: `0.973972920`
- Record-weighted Macro-F1: `0.976815792`
- Benign FNR: `0.0004951107810372571`
- Gafgyt FNR: `0.08512912604754574`
- Mirai FNR: `5.213924244747738e-05`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_00_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_00_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_00_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_00_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_00_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_00_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_00_compact_dnn_b0_seed2026\release_manifest.csv`
