# Locked YouTubePD External Audit Protocol

**Protocol freeze:** 9 July 2026, before any YouTubePD model performance was
computed. Metadata and file-integrity checks performed before this freeze are
listed below; they are not outcome analyses.

## Purpose and claim boundary

This analysis tests whether the Claim-Control Audit (CCA) can be executed on
independently collected raw facial video and whether an anatomical region
claim survives topology-matched and shortcut controls. It is not a clinical
validation study and does not estimate patient-level diagnostic performance.
YouTubePD provides public video clips but no verified participant identifier.
The unit of analysis is therefore the **video clip**, and all uncertainty will
be labelled clip-level.

## Frozen source state

- Local repository commit: `4379738`.
- Reconstructed clips: 248 of 282 valid spreadsheet records; 34 unavailable.
- Primary cohort source: `data_sheets/data_sheet.xlsx`, not the expanded
  negative-only sheet.
- Locally available primary records before technical quality control: 112
  clips (59 marked PD, 53 marked non-PD).
- Spreadsheet split among available records: train 63 (31/32), validation 25
  (17/8), and test 24 (11/13), where counts are PD/non-PD.
- The supplied text CSVs are retained only as a sensitivity analysis because
  they omit 19 available balanced-sheet records, mix spreadsheet validation
  records into both train and test, and repeat `video136` in `test.csv`.
- Repeated clips from one YouTube source URL will be grouped. No balanced-sheet
  source URL is repeated, but this rule remains active for sensitivity cohorts.

Frozen SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `data_sheet.xlsx` | `CB241BA68F3CAD5C3027EBBCAB11E13E5F214C8FD14C0E7615CA098B67D98DAC` |
| `NegSamples.xlsx` | `D02DF50B0D2DD901748F25095D58FEA9ED5C83114E8D37272D392E86756D0749` |
| `train.csv` | `02E7EB9A5BA91C40084B409CC55CE3C7D46CF29891A86411DB5083672BF8A67B` |
| `test.csv` | `119EDC1E99EEE929FC942A869F0DB04F8B9D4B06332688998422C21197C20BDC` |
| `region_video_annotations.pkl` | `1D2B561F78383284A909D43A1FB6767826C93C12C3D9CFF99DA2FFC40E23F661` |

## Primary cohort and endpoint

The primary endpoint is the spreadsheet `parkinson y/n` label in the balanced
sheet. The expanded negative-only sheet is excluded from the primary analysis
because source-sheet membership is perfectly associated with the negative
class and would create an avoidable collection shortcut.

The spreadsheet train split is used for model fitting and the validation split
for pipeline checks. After all settings pass the checks below, train and
validation are combined, the model is refit once, and the spreadsheet test
split is evaluated once. The test labels will not be used for exclusions,
thresholds, detector settings, hyperparameters, or support selection.

Severity labels among PD clips and the manually annotated frame-region labels
are exploratory secondary endpoints. Their results cannot replace the primary
binary audit and will be reported with their smaller denominators.

## Frozen video processing

1. Uniformly sample 24 timestamps between 5% and 95% of each clip duration.
2. Detect up to four faces with MediaPipe Face Landmarker in image mode.
3. Select a deterministic main-face track using face area and inter-frame
   centre/overlap continuity; labels are not available to the tracker.
4. Align each selected face from bilateral eye centres and mouth centre to a
   64 x 64 canonical frame, then downsample to 32 x 32.
5. Construct two clip summaries: the pixelwise temporal median (static
   appearance) and median absolute deviation (within-clip dynamics).
6. For the anatomical audit, robustly centre and scale each aligned frame
   before aggregation. Preserve an unnormalised branch for illumination and
   acquisition-shortcut sensitivity analyses.

A clip passes technical quality control when at least 12 of 24 sampled frames
contain a trackable face and the median aligned inter-eye distance is at least
12 pixels in the 64 x 64 frame. QC is label-blind and no clip will be manually
removed because its appearance is inconvenient for the hypothesis.

The external audit is downgraded to feasibility-only if fewer than 80% of the
112 primary clips pass QC or if the class-specific pass rates differ by more
than 10 percentage points. All failures and pass rates remain reportable.

## Frozen supports and model

The eight 32 x 32 supports already defined for PD-DBS are reused without
movement or resizing: upper brow/forehead, bilateral periocular, central
midface, bilateral cheek/zygomatic, perioral/mouth, and chin/mandible.

For each named rectangle, 64 unique translated rectangles of identical height,
width, topology and feature count are selected by the fixed seed `20260709 +
roi_index`. The named support and every translated support use the same
training records, preprocessing, feature representation and estimator.

Each support contributes its temporal-median and temporal-MAD pixels. The
locked estimator is a standardised L2 logistic regression (`C=1`, balanced
class weights, maximum 5,000 iterations). The estimator and all preprocessing
statistics are fitted only on the development records. No support-specific
hyperparameter tuning is allowed.

## Primary statistic and controls

For named region `r`, the primary location margin is:

`L(r) = AUROC(named r) - q0.975[AUROC(64 translated supports)]`

An anatomical-location claim requires `L(r) > 0`. AUROC, balanced accuracy,
and clip-level stratified-bootstrap 95% confidence intervals will be reported.
Support ranks are technical diagnostics, not patient-level P values.

The following controls are mandatory:

- full aligned face;
- complement of the predefined ROI union;
- aligned-image border;
- three fixed spatial thirds;
- exact-size scattered-pixel controls;
- coarse pooling and low-frequency retention;
- whole-frame acquisition context;
- context with the tracked face masked, plus the mask geometry alone;
- acquisition metadata only: year, frame size, frame rate, duration and file
  size, with missing values handled inside the development data.

The original author-provided region coordinates cover 220 local clips but were
computed after a separate person-centering and 20 fps pipeline. They will be
used only in a sensitivity analysis after geometric compatibility is verified;
they will not be projected directly onto the reconstructed raw frames.

## Sensitivity analyses

1. Supplied CSV assignment after removing the duplicated `video136` row.
2. Repeated stratified clip-level cross-validation across all QC-passing
   balanced-sheet clips; this is secondary to the locked test split.
3. Unnormalised aligned faces to quantify illumination dependence.
4. Static median and temporal MAD fitted separately.
5. Available YouTube source/uploader grouping, if metadata retrieval is
   sufficiently complete; absence of verified identity remains explicit.
6. Expanded negatives used only to test source-sheet shortcut sensitivity,
   never to inflate the primary external sample size.

## Interpretation rules

- High full-face AUROC alone establishes only clip-level discriminability.
- A named ROI is not anatomically specific unless it exceeds its translated
  97.5th percentile under the locked test and has the same sign in the repeated
  cross-validation sensitivity analysis.
- Strong context-only or metadata-only performance is evidence of collection
  shortcut susceptibility, not disease phenotype.
- Failure of named regions is a valid negative audit result and will not be
  hidden or replaced by a favourable post hoc region.
- YouTubePD cannot repair the absent PD-DBS patient identifiers, ethics number,
  consent documentation, or executable access mechanism.

## Reporting boundary

The manuscript may describe YouTubePD as independent public raw-video evidence
and PARK/UFNet as participant-level feature evidence where participant IDs are
available. It must not call YouTubePD a verified participant-independent cohort
or convert clip-level intervals into patient-level inference.
