# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_05_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:10:27.728427+00:00`
- Held-out cihaz: `Provision_PT_838_Security_Camera`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `15`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.998073414`
- Test fingerprint Macro-F1: `0.993238069`
- Validation-test farkı: `0.004835344`
- Test accuracy: `0.992367464`
- Record-weighted Macro-F1: `0.993375981`
- Benign FNR: `0.0005851375073142189`
- Gafgyt FNR: `0.020400390954130296`
- Mirai FNR: `0.0003517050708417863`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_05_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_05_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_05_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_05_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_05_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_05_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_05_compact_dnn_b0_seed2026\release_manifest.csv`
