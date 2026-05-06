# BenchmarkBridge v1 Analysis Plan

BenchmarkBridge v1 evaluates the frozen `D3_dfm_residual_gaze_only` model under
EyeBench-style internal split regimes. It is not a feature-engineering search, label
expansion, or neural baseline rerun.

Primary task: CopCo TYP classification. Auxiliary task: CopCo RCS regression if the
frozen comprehension target is available and sufficiently variable.

Split regimes:

- `unseen_reader`: test participants are disjoint from train participants.
- `unseen_text`: test speeches/texts are disjoint from train speeches/texts.
- `unseen_reader_and_text`: test participants and test texts are both disjoint from
  the training set.
- `text_balanced_unseen_reader`: participant-disjoint internal fold with deterministic
  text-exposure balancing by fold assignment.
- `leave_one_speech_out`: one speech/text held out.
- `participant_grouped_kfold`: deterministic participant-grouped k-fold.

Residualization is fit within each split/fold on training word rows only. Reader group,
participant ID, speech ID, text ID, labels, and targets are not residualizer predictors.
Primary models never use participant or speech identifiers as predictors.

Decision gates compare D3 against the request-specified CopCo TYP AhnCNN central values
for Unseen Reader and Unseen Reader + Text. Exact official EyeBench integration is
attempted but not required; if unavailable, results are reported as internal
EyeBench-style and benchmark-relative only.
