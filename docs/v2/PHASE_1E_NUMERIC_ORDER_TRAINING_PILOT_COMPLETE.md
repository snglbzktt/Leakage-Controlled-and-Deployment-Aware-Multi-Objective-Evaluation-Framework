# Phase 1E — Numeric-Order Training Pilot Complete

## Status

Locked and complete.

## Experimental design

- Numeric pipelines: Pipeline A and Pipeline B
- Models: TinyML-MLP and Compact-DNN
- Seeds: 42 and 123
- Total training runs: 8
- Matched comparisons: 4
- Primary selection metric: validation Macro-F1
- Test metrics used for selection: no
- Practical-equivalence margin: ±0.001

## Paired validation results

- `tinyml_mlp`, seed `42`: A=0.999603939220, B=0.999164696458, B-A=-0.000439242762
- `tinyml_mlp`, seed `123`: A=0.999575347028, B=0.999309343900, B-A=-0.000266003128
- `compact_dnn`, seed `42`: A=0.999398181277, B=0.999418462024, B-A=+0.000020280747
- `compact_dnn`, seed `123`: A=0.999478255248, B=0.999465502349, B-A=-0.000012752899

## Decision

- Median validation Macro-F1 delta B-A:
  `-0.000139378013`
- Pipeline A wins: `3/4`
- Pipeline B wins: `1/4`
- Locked decision:
  `practically_equivalent_retain_pipeline_a`
- Selected pipeline:
  `pipeline_a`

The median difference remains inside the pre-registered
±0.001 practical-equivalence margin. Pipeline A won three
of four matched comparisons, but its median advantage did
not reach the threshold required for a superiority decision.

The two orders are therefore treated as practically
equivalent. Pipeline A is retained for protocol continuity.
Test-set values were recorded but were not used to select
the numeric-processing order.
