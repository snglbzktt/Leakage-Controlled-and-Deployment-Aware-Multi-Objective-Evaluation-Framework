# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_07_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:29:16.271585+00:00`
- Held-out cihaz: `SimpleHome_XCS7_1002_WHT_Security_Camera`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `12`
- Seçilen epoch: `8`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.998750575`
- Test fingerprint Macro-F1: `0.987226238`
- Validation-test farkı: `0.011524337`
- Test accuracy: `0.990109167`
- Record-weighted Macro-F1: `0.984152564`
- Benign FNR: `0.017787023186237847`
- Gafgyt FNR: `0.016885727539817534`
- Mirai FNR: `0.0052664598790448285`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_07_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_07_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_07_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_07_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_07_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_07_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_07_compact_dnn_b0_seed2026\release_manifest.csv`
