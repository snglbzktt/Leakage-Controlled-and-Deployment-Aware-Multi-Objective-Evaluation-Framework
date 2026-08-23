# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_06_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:19:37.771915+00:00`
- Held-out cihaz: `Samsung_SNH_1011_N_Webcam`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `11`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.995878268`
- Test fingerprint Macro-F1: `0.993996951`
- Validation-test farkı: `0.001881317`
- Test accuracy: `0.986039380`
- Record-weighted Macro-F1: `0.994236333`
- Benign FNR: `0.000504449242317238`
- Gafgyt FNR: `0.01611315358817852`
- Mirai FNR: `None`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_06_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_06_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_06_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_06_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_06_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_06_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_06_compact_dnn_b0_seed2026\release_manifest.csv`
