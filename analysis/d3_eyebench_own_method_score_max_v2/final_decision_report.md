# Final Decision Report

- Final decision category: `d3_method_not_improved`
- Stop reason: `no_locked_candidate_improved_over_candidate_0000`
- Candidate count evaluated: 24
- Candidate_0000 reproduced: True
- Best candidate: `candidate_0000`
- Best not worse than candidate_0000: True
- Method evidence improved: False
- Official SOTA claimed: False
- Official leaderboard methods rerun: False
- W&B online API used: False
- Test-label tuning: False
- Synthetic predictions used: False
- Random predictions used: False

## Anchor Check
| split_name | metric | expected | actual | delta | tolerance | passed |
| --- | --- | --- | --- | --- | --- | --- |
| unseen_reader | balanced_accuracy | 0.7274 | 0.7274 | 0.0000 | 0.0010 | True |
| unseen_reader | roc_auc | 0.8085 | 0.8085 | 0.0000 | 0.0010 | True |
| unseen_text | balanced_accuracy | 0.7341 | 0.7341 | -0.0000 | 0.0010 | True |
| unseen_text | roc_auc | 0.8319 | 0.8319 | -0.0000 | 0.0010 | True |
| unseen_reader_and_text | balanced_accuracy | 0.6342 | 0.6342 | -0.0000 | 0.0010 | True |
| unseen_reader_and_text | roc_auc | 0.7154 | 0.7154 | -0.0000 | 0.0010 | True |
