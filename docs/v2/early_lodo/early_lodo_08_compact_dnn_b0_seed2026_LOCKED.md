# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_08_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:39:18.335259+00:00`
- Held-out cihaz: `SimpleHome_XCS7_1003_WHT_Security_Camera`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `12`
- Seçilen epoch: `9`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.997884211`
- Test fingerprint Macro-F1: `0.992487299`
- Validation-test farkı: `0.005396911`
- Test accuracy: `0.994646824`
- Record-weighted Macro-F1: `0.992753784`
- Benign FNR: `0.0006132917038358609`
- Gafgyt FNR: `0.009214904622277864`
- Mirai FNR: `0.0032416579264266015`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_08_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_08_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_08_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_08_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_08_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_08_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_08_compact_dnn_b0_seed2026\release_manifest.csv`
