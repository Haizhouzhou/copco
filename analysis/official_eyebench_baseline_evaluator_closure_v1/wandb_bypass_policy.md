# W&B Bypass Policy

- W&B API failure is not a scientific baseline failure.
- W&B online API is an orchestration and metadata retrieval layer.
- Missing W&B API credentials are recorded as `telemetry_orchestration_unavailable`.
- Missing W&B credentials must not by themselves set `baseline_reproduction_pass=false`.
- The baseline gate is evaluated from real local official-derived predictions and metrics.
- Fake, random, placeholder, manually typed, or diagnostic-only metrics cannot close the gate.
