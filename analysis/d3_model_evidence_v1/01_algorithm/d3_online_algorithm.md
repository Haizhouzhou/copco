# D3 Online Algorithm

Online D3 constructs prefix rows from observed evidence only. Prefix budgets include
word-count budgets and first-N text/trial budgets. At each prefix, the model outputs a
probability `p_t`; optional calibrators and thresholds are learned from inner data.

Accumulators include mean probability, logit mean, entropy/uncertainty weighting, and
learned meta-aggregation. V2 separates early, mid, late, full-evidence, and stopping
detector categories. Current interpretation: fixed-budget online D3 is a secondary
result; adaptive stopping is not ready.
