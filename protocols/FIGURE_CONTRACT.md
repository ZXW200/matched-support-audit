# Nature Portfolio Figure Contract V4

Every panel must state its experimental unit and must avoid implying biomarker
discovery, fixed-model faithfulness or causal anatomy. Five display items fit
the NMI Article limit and the NC Article format.

## Figure 1 | A matched spatial reference separates regional claims

**Conclusion:** predictive performance from a named support does not establish
that its location is exceptional.

- **a, Seven-claim ladder:** discrimination, support specificity, anatomical
  location, explanation faithfulness, shortcut profile, external evidence and
  clinical association.
- **b, Conditional estimand:** fixed dataset, representation, estimator and
  split; fresh training for the named rectangle and each translated rectangle.
- **c, Location margin:** named AUROC minus translated-support q97.5. Label the
  threshold as an operational descriptive rule, not a p value.
- **d, Evidence boundaries:** simulations, governance-limited PD-DBS images,
  YouTubePD clips and PARK participant features shown as non-equivalent tiers.

Do not draw an arrow from positive margin to biomarker. Show fixed-model
faithfulness as a separate test outside CCA.

## Figure 2 | Operating experiments distinguish simulated mechanisms

**Conclusion:** one specified CCA implementation returns different decisions
for injected local, border, distributed and misplaced signals.

- **a, Null repetitions:** 2/30 detections with Wilson interval and raw count.
- **b, Injected-effect curve:** detection proportions at effects 0.01-0.40,
  showing the denominator at every point.
- **c, Mechanism tasks:** local target, border, distributed cosine field and
  unknown sparse support.
- **d, Decision matrix:** target margin, border AUROC, 8 x 8 pooled AUROC and
  1 x 1 global-mean AUROC.

Use `operating experiments`, not `calibration`. State that synthetic independent
pixels do not determine error rates or power in facial data.

## Figure 3 | A legacy image matrix fails to support named locations

**Conclusion:** the supplied Class 0/Class 1 image split is highly
discriminative, but 0/8 named supports exceed the sampled spatial references.

- **a, Image-level discrimination:** full-image AUROC and post hoc cosine-
  similarity exclusion curve, including the number remaining at each threshold.
- **b, Matched locations:** named point, 64 translated-support values and q97.5
  for each rectangle.
- **c, Margins:** all eight named-minus-q97.5 values on a shared zero-centred
  axis. State `descriptive; no multiplicity-controlled p values`.
- **d, Distributed controls:** 32/64/128 random-pixel budgets, ROI-union
  complement, one-pixel border, three random thirds, 8 x 8 pooling and global
  mean.
- **e, Detector geometry:** observed exterior, box/landmark-only conditions and
  reassigned-mask distributions.

Panel footer: `image-level; patient identity unavailable; treatment-state label
mapping unresolved`. Do not label classes as pre-DBS or post-DBS.

## Figure 4 | A public-video stress test returns the same primary decision

**Conclusion:** the primary logistic estimator gives 0/8 location gates while
context and acquisition time remain predictive at clip level.

- **a, Reconstruction and QC:** 282 valid records, 248 reconstructed, 112 in the
  balanced cohort, 109 QC pass and 24 spreadsheet-test clips.
- **b, Primary location audit:** holdout and repeated-CV margins for eight
  regions. Explicitly state that repeated CV includes holdout clips.
- **c, Estimator sensitivity:** show the post hoc RBF-SVM mouth holdout margin
  (+0.0041) and its failed repeated-CV gate.
- **d, Competing supports:** aligned face, whole frame, face-masked context,
  middle third and acquisition metadata with clip-level intervals.
- **e, Collection time:** class-wise year distributions, year-only holdout AUROC
  and the exploratory 21-pair grouped-CV sensitivity.

Use no identifiable thumbnails. Do not call the cohort participant-independent,
external clinical validation or biological replication.

## Figure 5 | Experimental unit determines the permissible claim

**Conclusion:** participant-level released features can be associated with PD
labels without validating raw-image location.

- **a, PARK benchmark:** all-feature participant AUROC with participant
  bootstrap interval and training-label-shuffle distribution.
- **b, Feature families:** predefined groups labelled as sensitivity analyses,
  not raw anatomical regions.
- **c, Evidence matrix:** experimental unit, participant key, raw pixels,
  location test, shortcut controls, access and maximum permissible claim for
  PD-DBS, YouTubePD and PARK.
- **d, Final boundary:** discrimination present; primary location support absent
  in the two raw-data applications; faithfulness, causality and clinical utility
  not established by CCA.

Do not compare the three AUROCs as estimates of a common endpoint.

## Extended Data

1. All possible translation counts, sampled coordinates and overlap fractions.
2. Exact-size scattered-pixel controls and 50-draw random-budget distributions.
3. Static-only and dynamics-only YouTubePD location audits.
4. YouTubePD pooling, DCT and training-label controls.
5. Cross-split similarity and perceptual-hash distributions.
6. Acquisition metadata decomposition and year-matching caliper sensitivity.
7. Full RBF-SVM estimator sensitivity.
8. PARK participant aggregation, feature groups and calibration diagnostics.
9. Synthetic raw repetition tables and Wilson intervals.

## Source-data roots

- Figure 2: `outputs/nmi_synthetic_calibration/` and
  `outputs/nmi_audit_validation/`.
- Figure 3: `outputs/nmi_spatially_matched_audit/`,
  `outputs/nmi_exact_matched_audit/`, `outputs/hard_negative_random50/`,
  `outputs/data_qc/` and `outputs/nmi_detector_geometry_audit/`.
- Figure 4: `outputs/youtubepd_external_audit/`.
- Figure 5: `outputs/external/ufnet_participant_level_benchmark/`.

