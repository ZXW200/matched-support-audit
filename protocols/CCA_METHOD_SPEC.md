# Claim-Control Audit Specification

## Method Name

The proposed method is a **Claim-Control Audit (CCA)** for anatomical
interpretations of facial AI. CCA is not a diagnostic model and does not
establish a biomarker. It evaluates whether the evidence record satisfies the
preconditions required before an anatomical biomarker interpretation is made.

## Evidence Ladder

CCA separates claims that are often collapsed into one statement:

1. **Predictive signal**: the input-label task is discriminative under a locked
   evaluation protocol.
2. **Support specificity**: performance is not reproduced by arbitrary supports
   of the same size.
3. **Anatomical localisation**: named regions outperform spatially matched
   random and non-anatomical controls.
4. **Explanation faithfulness**: attribution or perturbation evidence passes
   model, label, seed, and remove-and-retrain sanity checks.
5. **Shortcut profile**: border, background, low-level, frequency, and split
   sensitivity are characterised.
6. **External evidence**: the audit behaviour transfers to an independent,
   subject-level dataset with compatible modality and labels.
7. **Clinical association**: the evidence relates to a patient-level clinical
   endpoint under an appropriate longitudinal or prospective design.

The first gate does not imply the second. The second does not imply the third.
The third does not establish clinical validity.

## Primary Audit Statistics

For a rectangular named support r, translated supports t with identical height
and width, and model seeds s, define the primary location margin:

`L(r) = mean_s AUROC(r,s) - q0.975_t[mean_s AUROC(t,s)]`

Positive L(r) is required for a named support to exceed the empirical location
null. Each translated support preserves topology and area, is fixed across
train, validation and test inputs, and is evaluated after retraining with the
same model seeds as the named support. Exact-size scattered pixels are a
separate pixel-budget control and are not the primary anatomical null.

For a support complement C, report:

`O(C) = AUROC(C)`

This is a support-retention statistic. It is not called background AUROC unless
C is produced by a validated face/background segmentation.

For a representation transformation T_k, report frequency retention relative to
the full image:

`F(T_k) = (AUROC(T_k) - 0.5) / (AUROC(full) - 0.5)`

All headline estimates should include test-sample bootstrap confidence
intervals where predictions are retained. Image-bootstrap intervals must be
labelled image-level when subject identifiers are unavailable. Matched-support
comparisons should include the empirical null distribution, point-estimate
difference and support rank. Support ranks are not patient-level P values.

## Falsifiability Requirements

CCA components are methodologically supported only to the extent that they
behave correctly on controlled tasks with known signal-generating supports:

- known ROI signal: the named ROI exceeds exact-size random supports;
- border shortcut: the named ROI fails while the border succeeds;
- distributed low-frequency signal: coarse representations retain signal and
  no single named support dominates the random null;
- unknown sparse support: a preselected named ROI does not become significant
  merely because some predictive pixels exist elsewhere.

These are unit tests for the audit logic. Repeated simulation is required to
estimate false-detection and sensitivity behaviour. Neither is a substitute
for clinical or multi-dataset method validation.

## Simulation Calibration

The topology-matched statistic was evaluated with a central 8 x 8 target and
99 disjoint translated 8 x 8 controls. Across 30 pure-null simulations, the
target exceeded the control 97.5th percentile 2 times (rate 0.0667; 95% Wilson
interval 0.0185-0.2132). Across 15 simulations per positive effect, detection
rates at effects 0.01, 0.02, 0.03, 0.04 and 0.05 were 0.067, 0.267, 0.533,
0.800 and 1.000. These are exploratory operating characteristics for one
data-generating process and one model family.

## PD-DBS Case Verdict

The method-validation outputs are in:

- `outputs/nmi_audit_validation/synthetic_audit_verdict.json`
- `outputs/nmi_audit_validation/synthetic_audit_summary.md`

The exact-size PD audit outputs are in:

- `outputs/nmi_exact_matched_audit/nmi_exact_matched_verdict.json`
- `outputs/nmi_exact_matched_audit/exact_roi_vs_random_summary.csv`
- `outputs/nmi_exact_matched_audit/nmi_exact_matched_summary.md`

The topology-matched PD audit outputs are in:

- `outputs/nmi_spatially_matched_audit/nmi_spatially_matched_verdict.json`
- `outputs/nmi_spatially_matched_audit/spatially_matched_roi_summary.csv`

The detector-geometry decomposition outputs are in:

- `outputs/nmi_detector_geometry_audit/nmi_detector_geometry_verdict.json`
- `outputs/nmi_detector_geometry_audit/detector_condition_summary.csv`
- `outputs/nmi_detector_geometry_audit/detector_mask_shuffle_summary.csv`

The label-free dynamic detector-box control is in:

- `outputs/nmi_dynamic_detector_control/nmi_dynamic_detector_verdict.json`
- `outputs/nmi_dynamic_detector_control/dynamic_detector_metrics.csv`

The PD-DBS image-level task produced the following audit verdict:

- 0 of 8 named ROIs exceeded the 97.5th percentile of identically shaped
  translated supports across three matched model seeds;
- the complement of the predefined ROI union reached AUROC 0.9174;
- a one-pixel border reached AUROC 0.9220;
- the dynamic detector-box interior reached AUROC 0.9701, while its exterior
  reached AUROC 0.8382;
- detector coordinates plus landmarks alone reached ensemble AUROC 0.8148;
- the three-seed exterior ensemble reached AUROC 0.8655, while global mask
  permutations averaged 0.7065;
- low and mid-frequency representations retained high performance, while the
  global-mean representation collapsed toward chance.

The complement of the predefined ROI union is not a true face/background
segmentation. The manuscript therefore uses `outside the predefined ROI union`
and reserves `non-face` for a future validated background mask.

## Locked YouTubePD Raw-Video Verdict

The external raw-video protocol was frozen before performance analysis in
`YOUTUBEPD_LOCKED_PROTOCOL_20260709.md`. It reused the same eight 32 x 32 named
supports and 64 topology-matched translations after label-blind face tracking,
alignment and clip-level temporal aggregation.

- 109/112 balanced-sheet clips passed the predeclared QC gate.
- All 24 locked test clips passed, including 11 PD and 13 non-PD clips.
- Aligned-face holdout AUROC was 0.9650, with a 95% clip-bootstrap interval of
  0.8881-1.0000.
- 0/8 named regions exceeded the translated q97.5 on the holdout.
- 0/8 also exceeded the translated q97.5 over 25 repeated-CV folds.
- Whole frame, face-masked context and acquisition metadata reached AUROC
  0.9301, 0.8601 and 0.8741.
- Source year alone reached AUROC 0.9021.

This result supports portability of the audit decision pattern from static
images to raw video. It is clip-level, not participant-level. The source
benchmark contains repeated public figures across years, and the local release
does not expose a verified participant key.

## PARK Feature-Level Boundary

The public UFNet/PARK release permits participant-level feature analysis but
not raw-image support testing. After participant aggregation, 42 smile features
reached AUROC 0.8560 in 116 official test participants, with a 95% participant
interval of 0.7842-0.9186. This demonstrates participant-level facial feature
association. It does not validate a named raw-image location.

## Boundaries

CCA can falsify an anatomical interpretation in a dataset. Passing CCA would
not prove that a signal is clinically causal, disease-specific,
treatment-responsive or useful. The PD-DBS matrix additionally lacks verifiable
provenance, ethics, consent and data-access documentation. YouTubePD lacks a
local participant key and contains strong collection-time structure. PARK is
participant-level but feature-only. These evidence boundaries cannot be
repaired by the audit itself.
