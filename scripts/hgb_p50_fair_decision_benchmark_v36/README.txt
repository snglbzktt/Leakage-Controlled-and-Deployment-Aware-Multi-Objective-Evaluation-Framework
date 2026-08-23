HGB vs TinyML-MLP P50-QAT fair class-decision CPU benchmark v3.6
Patch over v3.5: runtime-control context overhead is excluded from each timed call.
HGB threadpool_limits(1) is entered once around the benchmark loop.
TinyML torch.inference_mode() is entered once around the benchmark loop.
Both locked seed-2026 models are measured in five isolated processes.
Timed outputs are predicted class labels. Validation/test are not accessed.
