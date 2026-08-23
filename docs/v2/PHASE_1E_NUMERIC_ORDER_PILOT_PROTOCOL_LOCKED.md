# Phase 1E — Numeric-Order Training Pilot Protocol Locked

- Protocol version: numeric_order_training_pilot_v2_1
- Locked before training: yes
- Models: TinyML-MLP and Compact-DNN
- Numeric pipelines: A and B
- Seeds: 42 and 123
- Total runs: 8
- Task: family-3
- Device: CPU
- Primary selection metric: validation Macro-F1
- Test used for selection: no

## Run matrix

- `pipeline_a__tinyml_mlp__seed42` — learning rate 0.003
- `pipeline_a__tinyml_mlp__seed123` — learning rate 0.003
- `pipeline_a__compact_dnn__seed42` — learning rate 0.001
- `pipeline_a__compact_dnn__seed123` — learning rate 0.001
- `pipeline_b__tinyml_mlp__seed42` — learning rate 0.003
- `pipeline_b__tinyml_mlp__seed123` — learning rate 0.003
- `pipeline_b__compact_dnn__seed42` — learning rate 0.001
- `pipeline_b__compact_dnn__seed123` — learning rate 0.001

## Locked training settings

- AdamW
- Batch size 4096
- Maximum 15 epochs
- Early-stopping patience 4
- Early-stopping minimum delta 0.0002
- Weight decay 0.0001
- Train-only class weights
- Train-only scaler fitting
- One final test evaluation per validation-selected checkpoint

## Selection rule

Pipeline B is selected only when its median paired validation
Macro-F1 improvement is at least 0.001 and it wins at least three
of the four matched model-seed comparisons.

Pipeline A is selected under the symmetric negative condition.

Otherwise, the pipelines are treated as practically equivalent
and Pipeline A is retained. Test-set results are not used for this
selection.
