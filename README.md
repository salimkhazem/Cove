# COVE — distribution-free false-accept control for embedding-based verification

Official Implementation of _Identities, Not Pairs: Distribution-Free False-Accept Control for Embedding-Based Verification_.

A verification system accepts a pair of images as "same identity" when the similarity of their frozen embeddings exceeds a threshold. COVE replaces the threshold by a transductive test: rank the trial's similarity among all pairs of the calibration
identities augmented with the probe and the reference, and accept iff the rank is at most floor(alpha * N), N = C(m+2, 2). Under identity exchangeability its false-accept rate is at most floor(alpha*N)/N in finite samples.

## Quick start (one GPU, <= 24 GB)

    make setup          # environment from requirements.lock
    make test           # unit + validity + engine tests (< 1 min)
    make data features  # public datasets and cached embeddings
    make core toy       # main cell and synthetic runs
    make tables figures audit

`make reproduce` rebuilds the manuscript from the raw result records shipped in the
anonymized supplement; it does not rerun the GPU experiments. To rerun those, use
`make data features core toy breadth cohort adverse pac oracle` on the server, then
`make tables figures audit paper`. The full rerun requires substantially more compute.

## Layout

    src/cove/exact.py        exact integer similarity (the guarantee needs a fixed symmetric score)
    src/cove/conformal.py    COVE and every threshold rule
    src/cove/scores.py       cohort-normalized scores (CSLS, S-norm) valid inside COVE
    src/cove/evaluate.py     replicate engine (identity-uniform sampling = Assumption 1)
    src/cove/theory.py       zoo index, hubness
    scripts/                 feature extraction, experiment tiers, toy, mechanism runs
    tools/                   aggregation, figures, audit
    audits/                  claims-to-evidence map, AI-use log, reproduction spot check, compute
