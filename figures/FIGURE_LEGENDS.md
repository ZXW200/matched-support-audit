# Figure Legends

## Figure 1 | A matched spatial reference separates regional claims

**a,** Seven inferential claims and their evidence requirements. CCA evaluates support specificity and anatomical location, corresponding to claims 2 and 3. **b,** Two matched-control questions under a fixed dataset, representation, estimator and split. Exact-size scattered pixels test support specificity. Same-shape translated rectangles test anatomical location. Every support receives a fresh fit. **c,** Location margin, defined as named-support AUROC minus the translated-support q97.5. The threshold is a descriptive spatial reference. **d,** Resource-specific evidence tiers and maximum inference.

## Figure 2 | Operating experiments distinguish simulated mechanisms

**a,** Under a null synthetic task, CCA returned 2 detections in 30 independent repetitions. The proportion was 0.067 with a Wilson 95% interval of 0.018-0.213. The dashed line marks the nominal 0.025 reference implied by the operational 97.5th-percentile gate. **b,** Detection proportions for injected local effects of 0.01-0.40. Counts were 1/15, 4/15, 8/15, 12/15, 15/15, 15/15, 15/15 and 15/15, respectively. Shading denotes Wilson 95% intervals. **c,** Known synthetic mechanisms: a signal inside the named target, a border shortcut, a distributed low-frequency field and an unknown sparse support. **d,** Target-support, border, 8 x 8 pooled and global-mean AUROCs for the four tasks, with the resulting target-location gate. Every estimator was retrained for its tested support. Synthetic independent pixels provide operating checks for the specified model and data generator.

Source data: `fig2_synthetic_operating_curve.csv` and `fig2_mechanism_tasks.csv`.

## Figure 3 | A legacy image matrix reveals distributed image-level signal

**a,** Full-image discrimination and post hoc sensitivity to maximum cosine-similarity exclusion. Tick labels give the remaining held-out images. **b,** AUROCs for eight named supports (diamonds) and 64 same-shape translations per support (points). Horizontal lines denote translated-support q97.5 values. Each point is the mean of three fresh model fits using the supplied image split. **c,** Location margins. All eight were negative. **d,** Distributed and non-facial controls. For 32, 64 and 128 randomly sampled pixels, points and bars denote the mean and min-max range across 50 draws. Remaining controls are point estimates. The dashed line is the full-image AUROC. **e,** Detector-geometry conditions with image-bootstrap 95% intervals and mask-reassignment references summarised across 24 permutations. The experimental unit is the image. Numerical labels are reported as Class 0 and Class 1.

Source data: `fig3_full_image_metrics.csv`, `fig3_similarity_filter.csv`, `fig3_pd_roi_summary.csv`, `fig3_pd_translated_supports.csv`, `fig3_random_pixel_budgets.csv`, `fig3_distributed_controls.csv`, `fig3_detector_conditions.csv` and `fig3_detector_shuffle.csv`.

## Figure 4 | A public-video stress test returns the same primary location decision

**a,** Public reconstruction and locked cohort flow: 282 valid source records, 248 locally reconstructed clips, 112 clips in the balanced cohort, 109 clips passing quality control and 24 spreadsheet-defined test clips. Identifiable frames are excluded from display. **b,** Primary logistic CCA location margins for the spreadsheet holdout and repeated cross-validation. The result was 0/8 holdout gates and 0/8 joint gates. Repeated cross-validation includes spreadsheet-holdout clips. **c,** Post hoc RBF-SVM sensitivity. The perioral/mouth support exceeded the holdout reference by 0.004. Its repeated-cross-validation margin was -0.053, giving 1/8 holdout gates and 0/8 joint gates. **d,** Aligned-face, whole-frame, face-masked context, middle-third and acquisition-metadata controls. Diamonds and bars show spreadsheet-holdout AUROCs with clip-bootstrap 95% intervals. Open circles and bars show repeated-cross-validation means +/- s.d. **e,** Five-year source-date counts by numerical class. A year-only model achieved holdout AUROC 0.902 with a 95% interval of 0.748-1.000. An exploratory three-year-caliper analysis of 21 matched clip pairs yielded grouped-cross-validation AUROC 0.593 across 50 folds. The experimental unit is the video clip, and the spreadsheet provides no participant key.

Source data: `fig4_cohort_flow.csv`, `fig4_youtubepd_roi_summary.csv`, `fig4_rbf_roi_summary.csv`, `fig4_competing_supports.csv`, `fig4_year_distribution.csv`, `fig4_year_matched_controls.csv` and `fig4_collection_time_summary.csv`.

## Figure 5 | Experimental unit determines the permissible claim

**a,** PARK benchmark using released participant-aggregated smile features. The 42-feature model achieved participant-level test AUROC 0.856 with a participant-bootstrap 95% interval of 0.784-0.919. The test contained 116 participants, including 55 Class 1 and 61 Class 0 participants. Training-label shuffles had mean AUROC 0.504 and q97.5 0.650. **b,** Predefined feature-family sensitivity analyses with participant-bootstrap 95% intervals. **c,** Experimental-unit, data-access and executable-analysis boundaries for PD-DBS, YouTubePD and PARK. **d,** Observed evidence. Predictive discrimination was present across resources, and both raw spatial applications yielded 0/8 primary location gates. Fixed-model faithfulness, causal anatomy and clinical utility require separate evidence. The three AUROCs refer to resource-specific endpoints.

Source data: `fig5_park_benchmark.csv`, `fig5_park_feature_groups.csv` and `fig5_evidence_matrix.csv`.

## Extended Data Figure 1 | Fresh-source reconstruction changes discrimination while preserving the location verdict

**a,** Frozen-input and fresh-source YouTubePD reconstruction yields before quality control, after quality control and in the spreadsheet-defined test subset. **b,** Aligned-face AUROC changed from 0.965 to 0.827 across the full reconstructed cohorts. In the same 102 clip IDs and split, logistic AUROC changed from 0.916 to 0.867, while RBF-SVM AUROC was 0.902 for both encodings. **c,** The primary logistic estimator passed zero holdout and zero joint location gates under both reconstructions. The post hoc RBF-SVM passed one holdout gate and zero joint gates under both. The frozen input defines the primary analysis, and the fresh source provides a source-state sensitivity.

Source data: `extended_youtubepd_reconstruction_drift.csv` and `extended_youtubepd_common_cohort.csv`.

## Statistical and export notes

- AUROC is the area under the receiver operating characteristic curve.
- Wilson intervals are two-sided 95% binomial intervals.
- Bootstrap intervals are percentile intervals at the experimental unit stated in each legend.
- Cross-validation variability is the standard deviation across the reported folds or repeats.
- The CCA gate uses translated-support q97.5 as an operational spatial reference.
- Main figures are 183 mm wide. Editable SVG and PDF files, 600 dpi LZW-compressed TIFF files and 300 dpi PNG previews were generated with Python/matplotlib from aggregate source-data tables.
