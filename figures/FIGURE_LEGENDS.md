# Figure Legends

## Figure 1 | A matched spatial reference separates regional claims

**a,** Seven inferential claims that require distinct evidence. The Claim-Control Audit (CCA) addresses support specificity and anatomical location; it does not by itself test fixed-model explanation faithfulness, causal anatomy or clinical utility. **b,** Conditional estimand. With the dataset, representation, estimator and split held fixed, a fresh estimator is fitted to the named rectangle and to each same-shape translated rectangle. **c,** Operational location margin, defined as the named-support AUROC minus the 97.5th percentile of translated-support AUROCs. This is a descriptive gate, not a p value. **d,** Non-equivalent evidence tiers used in this study. A positive margin is not interpreted as a biomarker claim.

## Figure 2 | Operating experiments distinguish simulated mechanisms

**a,** Under a null synthetic task, the specified CCA implementation returned 2 detections in 30 independent repetitions (proportion 0.067; Wilson 95% interval, 0.018-0.213). The dashed line marks the nominal 0.025 reference implied by the operational 97.5th-percentile gate. **b,** Detection proportions for injected local effects of 0.01-0.40. Counts were 1/15, 4/15, 8/15, 12/15, 15/15, 15/15, 15/15 and 15/15, respectively; shading denotes Wilson 95% intervals. **c,** Known synthetic mechanisms: a signal inside the named target, a border shortcut, a distributed low-frequency field and an unknown sparse support. **d,** Target-support, border, 8 x 8 pooled and global-mean AUROCs for the four tasks, with the resulting target-location gate. Every estimator was retrained for its tested support. Synthetic independent pixels are operating experiments and do not determine error rates or power in facial data.

Source data: `fig2_synthetic_operating_curve.csv` and `fig2_mechanism_tasks.csv`.

## Figure 3 | A legacy image matrix fails to support named locations

**a,** Full-image discrimination and post hoc sensitivity to maximum cosine-similarity exclusion. Tick labels give the remaining held-out images. **b,** AUROCs for eight named rectangles (diamonds) and 64 same-shape translated rectangles per named region (points); horizontal lines denote translated-support q97.5 values. Each point is the mean of three fresh model fits using the supplied image split. **c,** Named-minus-q97.5 location margins. All eight margins were negative; the result is descriptive and no multiplicity-controlled p values were computed. **d,** Distributed and non-facial controls. For 32, 64 and 128 randomly sampled pixels, points and bars denote the mean and min-max range across 50 draws; remaining controls are point estimates. The dashed line is the full-image AUROC. **e,** Detector-geometry conditions with image-bootstrap 95% intervals and mask-reassignment references summarized across 24 permutations. Experimental unit, image. Patient identity was unavailable and the treatment-state mapping of the numerical class labels remained unresolved; classes are therefore not labelled as pre- or post-DBS.

Source data: `fig3_full_image_metrics.csv`, `fig3_similarity_filter.csv`, `fig3_pd_roi_summary.csv`, `fig3_pd_translated_supports.csv`, `fig3_random_pixel_budgets.csv`, `fig3_distributed_controls.csv`, `fig3_detector_conditions.csv` and `fig3_detector_shuffle.csv`.

## Figure 4 | A public-video stress test returns the same primary location decision

**a,** Public reconstruction and locked cohort flow: 282 valid source records, 248 locally reconstructed clips, 112 clips in the balanced cohort, 109 clips passing quality control and 24 spreadsheet-defined test clips. No identifiable frames are displayed. **b,** Primary logistic CCA location margins for the spreadsheet holdout and repeated cross-validation. None of eight named regions passed both gates; repeated cross-validation includes spreadsheet-holdout clips. **c,** Post hoc RBF-SVM sensitivity. The perioral/mouth rectangle exceeded the holdout reference by 0.004, but failed the repeated-cross-validation gate; one of eight regions passed the holdout gate and none passed both. **d,** Aligned-face, whole-frame, face-masked context, middle-third and acquisition-metadata controls. Diamonds and bars show spreadsheet-holdout AUROCs with clip-bootstrap 95% intervals; open circles and bars show repeated-cross-validation means +/- s.d. **e,** Five-year source-date counts by unresolved numerical class. A year-only model achieved holdout AUROC 0.902 (95% interval, 0.748-1.000); an exploratory three-year-caliper analysis of 21 matched clip pairs yielded grouped-cross-validation AUROC 0.593 across 50 folds. Experimental unit, video clip. This is not participant-independent external clinical validation.

Source data: `fig4_cohort_flow.csv`, `fig4_youtubepd_roi_summary.csv`, `fig4_rbf_roi_summary.csv`, `fig4_competing_supports.csv`, `fig4_year_distribution.csv`, `fig4_year_matched_controls.csv` and `fig4_collection_time_summary.csv`.

## Figure 5 | Experimental unit determines the permissible claim

**a,** PARK benchmark using released participant-aggregated smile features. The 42-feature model achieved participant-level test AUROC 0.856 (participant-bootstrap 95% interval, 0.784-0.919; 116 test participants, 55 class 1 and 61 class 0). Training-label shuffles had mean AUROC 0.504 and q97.5 0.650. **b,** Predefined feature-family sensitivity analyses with participant-bootstrap 95% intervals. These released feature groups are not raw-image anatomical regions. **c,** Experimental-unit, data-access and executable-analysis boundaries for PD-DBS, YouTubePD and PARK. **d,** Permissible inference: predictive discrimination is present in the tested resources, whereas the primary raw-data applications do not support the named locations; explanation faithfulness, causal anatomy and clinical utility are not established by CCA. The three resource-specific AUROCs do not estimate a common endpoint and are not compared as such.

Source data: `fig5_park_benchmark.csv`, `fig5_park_feature_groups.csv` and `fig5_evidence_matrix.csv`.

## Extended Data Figure 1 | Fresh-source reconstruction changes prediction but not the location verdict

**a,** Frozen-input and fresh-source YouTubePD reconstruction yields before quality control, after quality control and in the spreadsheet-defined test subset. **b,** Aligned-face AUROC changed from 0.965 to 0.827 across the full reconstructed cohorts. In the same 102 clip IDs and split, logistic AUROC changed from 0.916 to 0.867, whereas RBF-SVM AUROC was 0.902 for both encodings. **c,** The primary logistic estimator passed zero holdout and zero joint location gates under both reconstructions. The post hoc RBF-SVM passed one holdout gate and zero joint gates under both. This source-availability analysis was post hoc and does not replace the frozen-input primary analysis.

Source data: `extended_youtubepd_reconstruction_drift.csv` and `extended_youtubepd_common_cohort.csv`.

## Statistical and export notes

- AUROC is the area under the receiver operating characteristic curve.
- Wilson intervals are two-sided 95% binomial intervals.
- Bootstrap intervals are percentile intervals at the experimental unit stated in each legend.
- Cross-validation variability is the standard deviation across the reported folds or repeats.
- The CCA gate is an operational comparison with translated-support q97.5, not a null-hypothesis significance test.
- Main figures are 183 mm wide. Editable SVG and PDF files, 600 dpi LZW-compressed TIFF files and 300 dpi PNG previews were generated with Python/matplotlib from aggregate source-data tables.
