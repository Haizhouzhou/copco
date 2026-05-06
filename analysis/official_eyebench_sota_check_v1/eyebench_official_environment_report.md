# Official EyeBench Environment Report

- Environment name: `eyebench_official`
- Auto-create in CLI: False
- EyeBench imports: False
- Torch imports: False
- Status: `blocked_by_environment`

## Manual Create Attempts
- Exact command/status: {'command': 'mamba env create -n eyebench_official -f eyebench/environment.yml', 'status': 'failed', 'reason': 'strict-priority solve failed; libmamba reported incompatible packages and missing CUDA virtual package on the login node'}
- Flexible retry/status: {'command': 'CONDA_OVERRIDE_CUDA=12.4 mamba env create --channel-priority flexible -n eyebench_official -f eyebench/environment.yml', 'status': 'failed', 'reason': 'transaction selected packages but exited nonzero with libmamba callback invocation failure; resulting empty prefix was removed'}

## Live Import Attempt
- Command: `conda run -n eyebench_official bash -lc cd eyebench && python - <<'PY'
import sys
print('python', sys.version)
for name in ['beartype','pandas','numpy','sklearn','pyarrow','src']:
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'imported'))
    except Exception as exc:
        print(name, 'IMPORT_FAILED', repr(exc))
PY`
- Return code: 1
- stdout: ``
- stderr: `EnvironmentLocationNotFound: Not a conda environment: /home/haizhe/conda/envs/eyebench_official`

## Torch Attempt
- Return code: 1
- stdout: ``
- stderr: `EnvironmentLocationNotFound: Not a conda environment: /home/haizhe/conda/envs/eyebench_official`
