# Erken LODO HGB Bilimsel Koşu Kilidi

- Run ID: `early_lodo_07_hist_gradient_boosting_b0_seed2026`
- Kilit zamanı: `2026-08-05T22:31:32.672486+00:00`
- Held-out cihaz: `SimpleHome_XCS7_1002_WHT_Security_Camera`
- Model: `hist_gradient_boosting_b0`
- Seed: `2026`
- Tamamlanan iterasyon: `100`
- Validation seçim amacıyla kullanıldı: `False`

## Sonuçlar

- Validation Macro-F1: `0.999970284`
- Test fingerprint Macro-F1: `0.995889158`
- Validation-test farkı: `0.004081126`
- Test accuracy: `0.995507923`
- Record-weighted Macro-F1: `0.996038926`
- Benign FNR: `0.0002337322363500374`
- Gafgyt FNR: `0.012800027489991924`
- Mirai FNR: `0.00013638630837333997`

## Yerel karar

- Macro-F1 < 0.85: `False`
- Gafgyt FNR > 0.40: `False`
- Mirai FNR > 0.40: `False`
- Yerel ciddi kırılganlık: `False`

Bu karar yalnızca tek cihaz-model hücresine aittir. Genel erken
LODO değerlendirmesi kalan cihaz ve model koşuları tamamlandıktan
sonra yapılacaktır.

## Kilitli yapıtlar

- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\model.joblib`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\validation_predictions.npy`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\validation_probabilities.npy`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\test_predictions.npy`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\test_probabilities.npy`
- `results\v2\early_lodo\runs\early_lodo_07_hist_gradient_boosting_b0_seed2026\execution_state.json`
- `results\v2\early_lodo\locks\early_lodo_07_hist_gradient_boosting_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_07_hist_gradient_boosting_b0_seed2026\release_manifest.csv`
