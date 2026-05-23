# Stopping Policy Specification

`no_stop` consumes all available sequence evidence and is a full-evidence baseline, not
a stopping detector. Confidence, cost-sensitive, target-sensitivity, and
coverage-constrained stopping policies must learn thresholds from inner data. V2 found
that stopping reduced burden but did not preserve balanced accuracy, so stopping is not
ready.
