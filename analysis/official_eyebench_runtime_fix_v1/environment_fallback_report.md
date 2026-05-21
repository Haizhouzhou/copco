# Minimal Python 3.12 Runtime Fallback Report

- Primary CopCo env: `copco`
- Primary CopCo env imports ok: False
- Primary CopCo env Python 3.12 syntax blocker: True
- Minimal prefix: `/home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal`
- Prefix exists: True
- Imports ok: True
- Status: `ready`
- Classification: official-code/data/fold-compatible only if downstream data, fold, evaluator, and baseline gates pass.
- This is not an exact `environment.yml` reproduction.

## Primary CopCo Env Import
```text
{
  "env_name": "copco",
  "returncode": 1,
  "stdout": "3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]\nbeartype 0.20.2\npandas 3.0.2\npyarrow 24.0.0\nsklearn 1.8.0\neyebench src import ok",
  "stderr": "Traceback (most recent call last):\n  File \"<stdin>\", line 8, in <module>\n  File \"<frozen importlib._bootstrap>\", line 1176, in _find_and_load\n  File \"<frozen importlib._bootstrap>\", line 1147, in _find_and_load_unlocked\n  File \"<frozen importlib._bootstrap>\", line 690, in _load_unlocked\n  File \"<frozen importlib._bootstrap_external>\", line 936, in exec_module\n  File \"/home/haizhe/conda/envs/copco/lib/python3.11/site-packages/beartype/claw/_importlib/_clawimpload.py\", line 381, in get_code\n    return super().get_code(fullname)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"<frozen importlib._bootstrap_external>\", line 1074, in get_code\n  File \"/home/haizhe/conda/envs/copco/lib/python3.11/site-packages/beartype/claw/_importlib/_clawimpload.py\", line 455, in source_to_code\n    module_ast = compile(\n                 ^^^^^^^^\n  File \"/home/haizhe/copco/eyebench/src/data/utils.py\", line 295\n    f'Missing categories: {new_index.difference(groupby_fields)} in {\n    ^\nSyntaxError: unterminated string literal (detected at line 295)\n\nERROR conda.cli.main_run:execute(125): `conda run bash -lc cd eyebench && PYTHONPATH=$PWD:$PWD/src PYTHONNOUSERSITE=1 python - <<'PY'\nimport sys\nprint(sys.version)\nfor name in ['beartype','pandas','pyarrow','sklearn']:\n    mod = __import__(name)\n    print(name, getattr(mod, '__version__', 'imported'))\nimport src\nprint('eyebench src import ok')\nimport src.data.utils\nimport src.data.preprocessing.preprocess_data\nprint('eyebench preprocessing imports ok')\nPY` failed. (See above for error)",
  "imports_ok": false,
  "python312_syntax_blocker": true
}
```

## Log Tails
```text
al_py312_minimal\n\n  Updating specs:\n\n   - python=3.12\n   - pip\n\n\n  Package               Version  Build                 Channel           Size\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n  Install:\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n  + _openmp_mutex           4.5  20_gnu                conda-forge     Cached\n  + bzip2                 1.0.8  hda65f42_9            conda-forge     Cached\n  + ca-certificates   2026.5.20  hbd8a1cb_0            conda-forge      130kB\n  + ld_impl_linux-64     2.45.1  default_hbd61a6d_102  conda-forge     Cached\n  + libexpat              2.8.1  hecca717_0            conda-forge       77kB\n  + libffi                3.5.2  h3435931_0            conda-forge     Cached\n  + libgcc               15.2.0  he0feb66_19           conda-forge     Cached\n  + libgcc-ng            15.2.0  h69a702a_19           conda-forge     Cached\n  + libgomp              15.2.0  he0feb66_19           conda-forge     Cached\n  + liblzma               5.8.3  hb03c661_0            conda-forge     Cached\n  + libnsl                2.0.1  hb9d3cd8_1            conda-forge     Cached\n  + libsqlite            3.53.1  h0c1763c_0            conda-forge     Cached\n  + libuuid              2.42.1  h5347b49_0            conda-forge       40kB\n  + libxcrypt            4.4.36  hd590300_1            conda-forge     Cached\n  + libzlib               1.3.2  h25fd6f3_2            conda-forge     Cached\n  + ncurses                 6.6  hdb14827_0            conda-forge     Cached\n  + openssl               3.6.2  h35e630c_0            conda-forge     Cached\n  + packaging              26.2  pyhc364b38_0          conda-forge     Cached\n  + pip                  26.1.1  pyh8b19718_0          conda-forge     Cached\n  + python              3.12.13  hd63d673_0_cpython    conda-forge     Cached\n  + readline                8.3  h853b02a_0            conda-forge     Cached\n  + setuptools           82.0.1  pyh332efcf_0          conda-forge     Cached\n  + tk                   8.6.13  noxft_h366c992_103    conda-forge     Cached\n  + tzdata                2025c  hc9c84f9_1            conda-forge     Cached\n  + wheel                0.47.0  pyhd8ed1ab_0          conda-forge     Cached\n  + zstd                  1.5.7  hb78ec9c_6            conda-forge     Cached\n\n  Summary:\n\n  Install: 26 packages\n\n  Total download: 247kB\n\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n\n\nTransaction starting\nLinking libzlib-1.3.2-h25fd6f3_2\nLinking libgomp-15.2.0-he0feb66_19\nLinking zstd-1.5.7-hb78ec9c_6\nLinking _openmp_mutex-4.5-20_gnu\nLinking ld_impl_linux-64-2.45.1-default_hbd61a6d_102\nLinking libgcc-15.2.0-he0feb66_19\nLinking tk-8.6.13-noxft_h366c992_103\nLinking libsqlite-3.53.1-h0c1763c_0\nLinking libffi-3.5.2-h3435931_0\nLinking libgcc-ng-15.2.0-h69a702a_19\nLinking libuuid-2.42.1-h5347b49_0\nLinking libnsl-2.0.1-hb9d3cd8_1\nLinking liblzma-5.8.3-hb03c661_0\nLinking libexpat-2.8.1-hecca717_0\nLinking bzip2-1.0.8-hda65f42_9\nLinking ncurses-6.6-hdb14827_0\nLinking libxcrypt-4.4.36-hd590300_1\nLinking readline-8.3-h853b02a_0\nLinking tzdata-2025c-hc9c84f9_1\nLinking ca-certificates-2026.5.20-hbd8a1cb_0\nLinking openssl-3.6.2-h35e630c_0\nLinking python-3.12.13-hd63d673_0_cpython\nLinking pip-26.1.1-pyh8b19718_0\nLinking setuptools-82.0.1-pyh332efcf_0\nLinking packaging-26.2-pyhc364b38_0\nLinking wheel-0.47.0-pyhd8ed1ab_0\n\nTransaction finished\n\n\nTo activate this environment, use:\n\n    mamba activate /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal\n\nOr to execute a single command in this environment, use:\n\n    mamba run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal mycommand\n\n"
}
```

## Import Check
```text
{
  "prefix": "/home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal",
  "prefix_exists": true,
  "returncode": 0,
  "stdout": "3.12.13 | packaged by conda-forge | (main, Mar  5 2026, 16:50:00) [GCC 14.3.0]\nbeartype 0.20.2\npandas 2.2.3\npyarrow 24.0.0\nsklearn 1.8.0\nhydra 1.3.2\nsrc imported\ntext_metrics legacy api ok\neyebench preprocessing imports ok",
  "stderr": "",
  "imports_ok": true
}
```
