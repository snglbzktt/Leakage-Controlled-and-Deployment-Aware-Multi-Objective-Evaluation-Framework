# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_04_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:59:22.219272+00:00`
- Held-out cihaz: `Provision_PT_737E_Security_Camera`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `12`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.995238547`
- Test fingerprint Macro-F1: `0.962962680`
- Validation-test farkı: `0.032275867`
- Test accuracy: `0.960141662`
- Record-weighted Macro-F1: `0.964533858`
- Benign FNR: `0.0010150628070111837`
- Gafgyt FNR: `0.09554186390952987`
- Mirai FNR: `0.00433246943877434`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık sinyali: `False`

Bu değerlendirme yalnız tek cihaz-model hücresine aittir. Genel erken
LODO karar kapısı, kalan cihaz ve model koşuları tamamlanmadan
sonuçlandırılamaz.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_04_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_04_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_04_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_04_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_04_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_04_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_04_compact_dnn_b0_seed2026\release_manifest.csv`
