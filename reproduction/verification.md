# Clean-environment verification

Overall: **PASS** 
(41/41 checks).

| Check | Observed | Expected | Pass |
|---|---:|---:|:---:|
| synthetic.0.0.detections | 2 | 2 | yes |
| synthetic.0.0.repetitions | 30 | 30 | yes |
| synthetic.0.01.detections | 1 | 1 | yes |
| synthetic.0.01.repetitions | 15 | 15 | yes |
| synthetic.0.02.detections | 4 | 4 | yes |
| synthetic.0.02.repetitions | 15 | 15 | yes |
| synthetic.0.03.detections | 8 | 8 | yes |
| synthetic.0.03.repetitions | 15 | 15 | yes |
| synthetic.0.04.detections | 12 | 12 | yes |
| synthetic.0.04.repetitions | 15 | 15 | yes |
| synthetic.0.05.detections | 15 | 15 | yes |
| synthetic.0.05.repetitions | 15 | 15 | yes |
| synthetic.0.1.detections | 15 | 15 | yes |
| synthetic.0.1.repetitions | 15 | 15 | yes |
| synthetic.0.2.detections | 15 | 15 | yes |
| synthetic.0.2.repetitions | 15 | 15 | yes |
| synthetic.0.4.detections | 15 | 15 | yes |
| synthetic.0.4.repetitions | 15 | 15 | yes |
| mechanism.roi_localised.target_roi_auroc | 1.0 | 1.0 | yes |
| mechanism.roi_localised.random_exact_q975 | 0.9276438888888888 | 0.9276438888888888 | yes |
| mechanism.border_shortcut.target_roi_auroc | 0.5004972222222223 | 0.5004972222222223 | yes |
| mechanism.border_shortcut.border_auroc | 1.0 | 1.0 | yes |
| mechanism.distributed_low_frequency.pool_8x8_auroc | 1.0 | 1.0 | yes |
| mechanism.distributed_low_frequency.pool_1x1_auroc | 0.5205888888888889 | 0.5205888888888889 | yes |
| youtubepd.primary_cohort_clips_before_qc | 112 | 112 | yes |
| youtubepd.clips_after_qc | 109 | 109 | yes |
| youtubepd.test_clips | 24 | 24 | yes |
| youtubepd.named_rois_passing_holdout_location_gate | 0 | 0 | yes |
| youtubepd.named_rois_passing_both_location_gates | 0 | 0 | yes |
| youtubepd.aligned_face_holdout_auroc | 0.965034965034965 | 0.965034965034965 | yes |
| youtubepd.whole_frame_holdout_auroc | 0.93006993006993 | 0.93006993006993 | yes |
| youtubepd.face_masked_context_holdout_auroc | 0.8601398601398602 | 0.8601398601398602 | yes |
| youtubepd.metadata_only_holdout_auroc | 0.8741258741258742 | 0.8741258741258742 | yes |
| youtubepd_rbf.full_face_holdout_auroc | 0.916083916083916 | 0.916083916083916 | yes |
| youtubepd_rbf.named_rois_passing_holdout_gate | 1 | 1 | yes |
| youtubepd_rbf.named_rois_passing_both_gates | 0 | 0 | yes |
| park.train_plus_dev_participants | 1245 | 1245 | yes |
| park.test_participants | 116 | 116 | yes |
| park.participant_auroc | 0.8560357675111774 | 0.8560357675111774 | yes |
| park.participant_auprc | 0.8186813462447392 | 0.8186813462447392 | yes |
| park.participant_balanced_accuracy | 0.7970193740685544 | 0.7970193740685544 | yes |

Verification covers deterministic analysis from the available third-party inputs. It does not guarantee future availability or byte identity of remote videos.
