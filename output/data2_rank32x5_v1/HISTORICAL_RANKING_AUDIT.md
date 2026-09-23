# Historical ranking audit

A historical untracked `feat_input2/run_data2_nextstep.py` and `output/data2_nextstep/` were found. They contain one partial `output/data2_nextstep/ranking/raw_scores/holdout_I0002.csv` and diagnostics, not a complete three-fold result. Its protocol is not the present Rank32x5 protocol: it uses all eligible pairs without the per-positive cap, a single ranker, no five independent subsamples, and no required coefficient/overlap diagnostics. It was therefore retained untouched as historical evidence and was not used as the formal result.
