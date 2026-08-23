# Erken LODO HGB Bilimsel Koşu Kilidi

- Run ID: `early_lodo_04_hist_gradient_boosting_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:01:38.124565+00:00`
- Held-out cihaz: `Provision_PT_737E_Security_Camera`
- Model: `hist_gradient_boosting_b0`
- Seed: `2026`
- Tamamlanan iterasyon: `100`
- Validation seçim amacıyla kullanıldı: `False`

## Sonuçlar

- Validation Macro-F1: `0.999996642`
- Test fingerprint Macro-F1: `0.924677990`
- Validation-test farkı: `0.075318652`
- Test accuracy: `0.920704202`
- Record-weighted Macro-F1: `0.928190131`
- Benign FNR: `0.0`
- Gafgyt FNR: `0.200668230924944`
- Mirai FNR: `0.0011811655695970276`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık: `False`

Bu karar yalnızca tek cihaz-model hücresine aittir. Genel erken
LODO değerlendirmesi kalan cihaz ve model koşuları tamamlandıktan
sonra yapılacaktır.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\model.joblib`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\validation_predictions.npy`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\validation_probabilities.npy`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\test_predictions.npy`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\test_probabilities.npy`
- `results\v2\early_lodo\runs\early_lodo_04_hist_gradient_boosting_b0_seed2026\execution_state.json`
- `results\v2\early_lodo\locks\early_lodo_04_hist_gradient_boosting_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_04_hist_gradient_boosting_b0_seed2026\release_manifest.csv`
