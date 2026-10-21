# Gate G2 decision record (2026-10-20, before any Center 2 outcome was read)

Evidence: fold-0 pilot, both segmenters, 20 epochs, scored on the 12 held-out fold-0 Center 1 patients (1,915 frames, D1 1,598, D2 317; 384-grid IoU inside the eroded FOV). Scripts: `work/g2_check.py` (output `work/logs/g2_fold0_pilot_summary.json`, `g2_fold0_pilot_frames.csv`), training curves `work/models/train_log_*_fold0_pilot.csv`. No Center 2 data were opened.

Findings
- Inner-validation IoU plateaus under the cosine schedule: U-Net 0.942 (epoch 14) to 0.944 (epoch 19-20); SegFormer-B0 0.938 (epoch 15) to 0.940 (epoch 20). Gain over the last five epochs is at most 0.003 IoU.
- Held-out median IoU 0.974 (U-Net) and 0.971 (SegFormer); 5th percentile 0.88; the failure tail is thin.
- Events by threshold (U-Net / SegFormer, share of 1,915 frames; patients with an event of 12):
  IoU < 0.75: 20 / 19 events (1.0%), 7 / 7 patients;
  IoU < 0.80: 32 / 30 (1.6-1.7%), 9 / 8;
  IoU < 0.85: 55 / 59 (2.9-3.1%), 11 / 11;
  IoU < 0.90: 126 / 135 (6.6-7.1%), 12 / 12.
  One patient (P63) holds 1,159 of the 1,915 frames; pooled statistics rely on the patient-clustered bootstrap.
- Ensemble disagreement and IoU are strongly related (Spearman -0.79 and -0.83), so the comparator is meaningful.

Decisions (frozen in `config/analysis_lock.yaml`)
1. Epochs: 20 for U-Net and 20 for SegFormer-B0 (curves have flattened; no gain expected from more).
2. Failure threshold revised from 0.75 to 0.90 (the plan allows revision at G2 only). At 0.75 about 1% of frames fail, which projects to roughly 100 Center 1 events and about 11 Center 2 events: too few to fit a 33-coefficient penalised gate stably and below the plan's own 30-event floor for external binary metrics. At 0.90 the projection is roughly 630 Center 1 events spread over every patient and about 70 Center 2 events if the rate transfers. 0.90 is also the cut the perturbation experiment uses for "intact" frames. The primary endpoint (external AURC on continuous risk 1 - IoU) is unaffected. The perturbation collapse rule (`perturbation.rule_first_below_iou`, median IoU 0.75) is a separate descriptive criterion and stays.
3. Non-inferiority margin: none (null). Mean risk (random AURC) is 0.040 to 0.043 and the oracle AURC 0.018 to 0.020 in the pilot, so the default margin of 0.01 would be nearly half of the whole random-to-oracle range and would make any non-inferiority claim nearly empty. Comparisons against entropy and ensemble disagreement are therefore descriptive.
4. Entropy baseline (fov vs roi) is chosen later by `R/04a_gate_fit_lock.R` on Center 1 out-of-fold data only.
