# Matched-support audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21292743.svg)](https://doi.org/10.5281/zenodo.21292743)

This repository contains the public, executable branches and aggregate source
data for *Matched-support auditing tests anatomical specificity in facial
artificial intelligence*.

The matched-support audit asks a conditional question: under a fixed
dataset, representation, estimator and split, does a model retrained on a named
support outperform models retrained on shape- and area-matched translations?
The audit evaluates dataset-level regional predictive sufficiency. It is not a
fixed-model attribution method, a causal anatomy test or clinical validation.

## Public scope

Included:

- synthetic operating-characteristic and mechanism experiments;
- a clip-level YouTubePD reconstruction and audit pipeline;
- a participant-level UFNet/PARK feature benchmark;
- aggregate source data, figure scripts and publication exports;
- frozen protocols, dependency versions and clean-environment verification.

Excluded:

- the governance-limited PD-DBS image matrix and all derived sample-level data;
- reconstructed YouTube videos and identifiable facial images;
- UFNet/PARK raw participant videos;
- credentials, local paths and private submission materials.

The PD-DBS results reported in the manuscript are a non-essential legacy stress
test and cannot be independently rerun from this public release.

## Reproduction

Create an isolated Python 3.13 environment:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-reproduction.txt
```

Fetch the fixed third-party repositories and the MediaPipe model:

```powershell
.venv\Scripts\python scripts/fetch_third_party.py
```

YouTubePD does not provide redistributable raw clips. Reconstruct the balanced
cohort from the source URLs where they remain available:

```powershell
.venv\Scripts\python scripts/download_youtubepd_clips.py `
  --youtube-root external/YouTubePD-data --sheet balanced --allow-failures
```

Run all public branches:

```powershell
.venv\Scripts\python scripts/run_public_reproduction.py
.venv\Scripts\python scripts/verify_reproduction.py
```

Remote videos can disappear or change. The verification report distinguishes
code/environment reproducibility from source-video availability and never
interprets clips as participant-independent observations.

The frozen synthetic result consists of two runs (`0,0.05,0.10,0.20,0.40` and
`0.01,0.02,0.03,0.04`) because the original script bound dataset seeds to each
effect's list position. The runner preserves those commands and combines them;
it does not silently substitute a different one-pass simulation.

## Figures

All visual output is generated exclusively with Python/Matplotlib from the
bundled aggregate source-data tables:

```powershell
.venv\Scripts\python scripts/make_publication_figures.py
.venv\Scripts\python scripts/qa_publication_figures.py
```

Each main figure is exported as editable SVG, PDF, 600-dpi TIFF and PNG preview.
Quantitative panels map to CSV files under `source_data/`. Figure legends and
statistical definitions are in `figures/FIGURE_LEGENDS.md`; the automated export
audit is in `figures/FIGURE_QA_REPORT.md` (114/114 checks passed in the release
environment).

`scripts/build_source_data.py` is a maintainer-side aggregation script. It
requires the restricted PD-DBS aggregate-output root and is not needed to render
or verify the released figures.

## Clean-room audit

The public branches were executed in a new Python 3.13 environment without
system site packages. Independent verification passed 41/41 expected-result
checks. The report, realized dependency set, fixed third-party manifest and
fresh-source reconstruction sensitivity are under `reproduction/`.

## Citation

Wang, Z. *Matched-support auditing tests anatomical specificity in facial
artificial intelligence*, version 1.0.3. Zenodo (2026).
[https://doi.org/10.5281/zenodo.21292743](https://doi.org/10.5281/zenodo.21292743).

## Third-party resources

- YouTubePD commit `43797386a65ffb58db53628b90ef8e8f35512e0d`.
- UFNet commit `5ece2c65ba184faccf6c8cdccdc03132427c464b`.
- MediaPipe Face Landmarker SHA-256
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

YouTubePD files are obtained from the source repository because that repository
does not contain an explicit licence file. This release does not redistribute
those files or videos.

## Licence

Python source code under `src/` and `scripts/` is available under the MIT
License (`LICENSE-CODE`). Aggregate source data, figures, protocols,
documentation and reproduction reports are available under CC BY 4.0
(`LICENSE-CONTENT`). Third-party repositories, models, spreadsheets and remote
videos are excluded from these grants and retain their original terms.

## Evidence boundary

The empirical q97.5 support gate is a descriptive spatial reference, not a
calibrated p value or family-wise error-controlled test. Failure to cross the
gate is not an equivalence result. All YouTubePD estimates are clip-level, and
PARK raw-image location is not tested because only extracted features are
public.
