# EyeBench Environment Report

- Environment name: `eyebench`
- Python version/status: 3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]
- EyeBench imports: False
- CLI/scripts runnable: False
- CUDA required: not for this adapter; EyeBench environment declares PyTorch CUDA for neural baselines.
- WandB login required: official train/sweep scripts use WandB; this adapter does not.
- Official data download possible: False

## Environment Create Attempt
- Command: `mamba env create -f eyebench/environment.yml`
- Return code: 1
- stderr: `critical libmamba Aborting.`

## Import Attempt
- Command: `conda run -n eyebench python -c import sys; print(sys.version); import pandas; print('pandas', pandas.__version__); import src; print('src_import_ok')`
- Return code: 1
- stdout: `3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]
pandas 3.0.2`
- stderr: `Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/haizhe/copco/eyebench/src/__init__.py", line 3, in <module>
    from beartype import BeartypeConf  # <-- this isn't your fault
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'beartype'

ERROR conda.cli.main_run:execute(125): `conda run python -c import sys; print(sys.version); import pandas; print('pandas', pandas.__version__); import src; print('src_import_ok')` failed. (See above for error)`
