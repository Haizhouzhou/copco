# Rewrite AGENTS Compute Policy

## Task Summary

Rewrite `AGENTS.md` to emphasize environment boot rules, tested Slurm/resource
profiles, heavy-compute policy, AI-run logging, and git safety for the `copco`
research checkpoint repository.

## Plan

1. Inspect repository state and current `AGENTS.md`.
2. Inspect local/global resource policy sources that future agents should reuse.
3. Rewrite `AGENTS.md` without STN-specific content or phase-gate language.
4. Update the AI run log index.
5. Run lightweight validation and git safety checks.
6. Stage explicit safe paths, commit, and push.

## Files Inspected

- `AGENTS.md`
- `~/.codex/AGENTS.md`
- `~/bin/claim_best_immediate_resource.sh`
- `scripts/validate_env.py`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-02_0310_initialize_repo.md`

## Files Modified

- `AGENTS.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-02_0333_rewrite_agents_compute_policy.md`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `git branch --show-current`
- `git remote -v`
- `sed -n '1,260p' AGENTS.md`
- `date +%Y-%m-%d_%H%M`
- `sed -n '1,220p' ~/.codex/AGENTS.md`
- `sed -n '1,220p' ~/bin/claim_best_immediate_resource.sh`
- `find scripts -maxdepth 2 -type f`
- `find . -path './.git' -prune -o \\( -name '*.slurm' -o -name '*.sbatch' -o -name 'submit*.sh' -o -name '*slurm*.sh' \\) -type f`
- `find logs -maxdepth 3 -type f`
- `sed -n '1,320p' AGENTS.md`
- `rg -n "Phase|STN|LFP|beta|MEG|Brian2|Brian2CUDA|DYNAP|adaptive DBS|beta-burst|\\.fif|balanced" AGENTS.md`
- `python scripts/validate_env.py`
- `python -m pytest tests/ -q`
- `git diff --check`
- `python -m ruff check .`
- `git status --short`
- `git diff --stat`
- `git add AGENTS.md logs/ai_runs/INDEX.md logs/ai_runs/2026-05-02_0333_rewrite_agents_compute_policy.md`
- `git diff --stat --cached`
- `git diff --cached --check`
- staged-file size check using `git cat-file -s`

## Validation Results

- `python scripts/validate_env.py`: passed.
- `python -m pytest tests/ -q`: passed, 1 test.
- `git diff --check`: passed.
- Removed phase-gate and old STN-specific content. The only inspected keyword match is
  the required warning against generic `balanced` guesses.
- `python -m ruff check .`: not run successfully because Ruff is not installed in the
  current Python environment.
- Pre-commit staged diff check: passed.
- Staged large-file check: passed; staged files are small text files.

## Commit / Push Status

This run log is committed with the task changes. The final assistant response records
the exact commit hash and push result because embedding the final commit hash in this
file would change the hash.

## Final Response Summary

Will report files changed, validation outcomes, commit hash, push status, and unresolved
issues.
