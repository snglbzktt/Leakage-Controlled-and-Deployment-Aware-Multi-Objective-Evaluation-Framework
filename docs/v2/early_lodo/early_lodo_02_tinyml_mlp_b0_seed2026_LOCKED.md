# Erken LODO Bilimsel Koşu Kilidi

- Run ID: `early_lodo_02_tinyml_mlp_b0_seed2026`
- Kilit zamanı: `2026-08-05T21:22:14.679441+00:00`
- Held-out cihaz: `Ennio_Doorbell`
- Model: `tinyml_mlp_b0`
- Seed: `2026`
- Tamamlanan epoch: `15`
- Seçilen epoch: `14`
- Erken durdurma: `False`

## Sonuçlar

- Validation Macro-F1: `0.996887384`
- Test fingerprint Macro-F1: `0.992445435`
- Validation-test farkı: `0.004441948`
- Test accuracy: `0.982395105`
- Record-weighted Macro-F1: `0.992704158`
- Benign FNR: `0.0006495340299350466`
- Gafgyt FNR: `0.01958514326047449`
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

- `results\v2\early_lodo\runs\early_lodo_02_tinyml_mlp_b0_seed2026\run_result.json`
- `results\v2\early_lodo\runs\early_lodo_02_tinyml_mlp_b0_seed2026\history.csv`
- `results\v2\early_lodo\runs\early_lodo_02_tinyml_mlp_b0_seed2026\best_model.pt`
- `results\v2\early_lodo\runs\early_lodo_02_tinyml_mlp_b0_seed2026\checkpoint.pt`
- `results\v2\early_lodo\runs\early_lodo_02_tinyml_mlp_b0_seed2026\run_report.txt`
- `results\v2\early_lodo\locks\early_lodo_02_tinyml_mlp_b0_seed2026\lock_summary.json`
- `results\v2\early_lodo\locks\early_lodo_02_tinyml_mlp_b0_seed2026\release_manifest.csv`
