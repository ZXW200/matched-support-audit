# Clean-room reproduction report

## Scope

The public synthetic, YouTubePD and PARK branches were copied into a new
directory and executed from a newly created Python 3.13 virtual environment
without system site packages. Dependencies were installed from
`requirements-reproduction.txt`; the realized environment is recorded in
`pip-freeze.txt`. Third-party repositories and the face-landmarker model were
fetched at the commits and hash recorded in `third_party_manifest.json`.

The governance-limited PD-DBS matrix was not copied into the clean room and is
not part of the executable public verification.

## Result

The corrected public runner completed every analysis command with exit code 0.
Independent result verification passed **41/41 checks**. Exact observed and
expected values are recorded in `verification.json` and `verification.md`.

The checks cover:

- null and non-null synthetic operating repetitions;
- known local, border and distributed mechanism tasks;
- the frozen 112-clip YouTubePD cohort, QC, primary logistic location decision,
  competing supports and post hoc RBF-SVM sensitivity;
- the PARK participant split and primary participant-level metrics.

The publication-figure renderer uses fixed metadata and canonical SVG object
identifiers. Two consecutive render-and-QA passes produced byte-identical
hashes for all 29 files in the figure bundle; the final format and source-data
audit passed 114/114 checks.

## Issues exposed by the clean-room run

The first import attempt showed that the NumPy MLP module imported two data
loader modules that had not been included in the initial package selection.
Those source modules were added; they do not include restricted data.

The synthetic implementation binds random seeds to an effect's list position.
The manuscript values had therefore been produced by one main-effect run and
one low-effect run. A one-pass effect list changes the low-effect seeds. The
public runner now executes the two historical commands explicitly and combines
their aggregate outputs. `synthetic_seed_correction_run.json` records this
decision; the final verifier then passed all synthetic checks exactly.

## Remote-source drift audit

Remote videos are mutable and can disappear. A fresh reconstruction produced
117 balanced clips (113 passing QC; 26 spreadsheet-test clips), compared with
112, 109 and 24 in the frozen primary input. Aligned-face holdout AUROC changed
from 0.965 to 0.827, but the primary location result remained zero holdout gates
and zero joint gates. The post hoc RBF-SVM remained one holdout gate and zero
joint gates.

For the 102 clip IDs present and QC-valid in both encodings, logistic AUROC
changed from 0.916 to 0.867, while RBF-SVM AUROC was 0.902 for both. Location
gate counts were unchanged. Aggregate results are in
`common_cohort_comparison.csv` and `fresh_source_audit_summary.json`.

This is a post hoc source-availability audit. It demonstrates that predictive
performance is sensitive to reconstruction state while the tested location
decision was stable in this instance. It does not guarantee future video
availability or byte-identical reconstruction.
