# Online Accumulation Specification

Online D3 can aggregate prefix probabilities using simple mean probability, cumulative
logit mean, entropy/uncertainty weighting, reliability weighting, or a learned
meta-aggregator trained on inner-validation predictions. V1 selected a strong
learned-meta no-stop accumulator; v2 separates that from true fixed-budget online rows.
