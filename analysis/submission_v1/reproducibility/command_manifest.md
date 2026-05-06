# Command Manifest

- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir results/submission_v1_<timestamp>`
- `conda run -n copco copco-validate-submission-package --config configs/submission_v1.yaml --output-dir results/submission_v1_<timestamp>`
