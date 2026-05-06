# Command Manifest

- Validate env: `conda run -n copco python scripts/validate_env.py`
- Validate Label Release: `conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041`
- Validate Phase 4: `conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715`
- Run AutoResearch: `conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir results/autoresearch_v1_<timestamp>`
