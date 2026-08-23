# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_01_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T20:12:58.759665+00:00`
- Held-out cihaz: `Ecobee_Thermostat`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `13`
- Seçilen epoch: `13`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.998524948`
- Test fingerprint Macro-F1: `0.990265033`
- Validation-test farkı: `0.008259915`
- Test accuracy: `0.993423914`
- Record-weighted Macro-F1: `0.990390551`
- Benign FNR: `0.0005339028296849973`
- Gafgyt FNR: `0.01446842794320051`
- Mirai FNR: `0.002140069083616951`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_01_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_01_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_01_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_01_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_01_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_01_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_01_compact_dnn_b0_seed2026\release_manifest.csv`
