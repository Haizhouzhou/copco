# Environment Cleanup Report

- Named env removal attempted: False
- Reason: revised runtime rule uses copco env first and does not remove/create full EyeBench env unless necessary
- `eyebench_official` still listed: True
- legacy `eyebench` still listed: True
- Status: `complete_or_quarantined`

## Live Cleanup Processes
```text
3978513 3978382       00:00 S+   bash -lc ps -o pid,ppid,etime,stat,cmd -u "$USER" | rg 'conda env remove|mamba env create|/usr/bin/rsync -a --force --delete' || true
3978528 3978513       00:00 S+   rg conda env remove|mamba env create|/usr/bin/rsync -a --force --delete
```
