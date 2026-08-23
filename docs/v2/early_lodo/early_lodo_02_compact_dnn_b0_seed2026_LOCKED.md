# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_02_compact_dnn_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:27:22.125848+00:00`
- Held-out cihaz: `Ennio_Doorbell`
- Model: `compact_dnn_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `12`
- Erken durdurma: `True`

## Sonuçlar

- Validation Macro-F1: `0.996619587`
- Test fingerprint Macro-F1: `0.993244568`
- Validation-test farkı: `0.003375018`
- Test accuracy: `0.985829846`
- Record-weighted Macro-F1: `0.993478255`
- Benign FNR: `0.0005648121999435188`
- Gafgyt FNR: `0.015759146934750272`
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

- `results\v2\early_lodo\runs\early_lodo_02_compact_dnn_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_02_compact_dnn_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_02_compact_dnn_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_02_compact_dnn_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_02_compact_dnn_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_02_compact_dnn_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_02_compact_dnn_b0_seed2026\release_manifest.csv`
