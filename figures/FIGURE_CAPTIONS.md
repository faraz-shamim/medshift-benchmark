# Figure captions and selection guide

F1: Executed study flow. Counts are image counts, not claimed independent patients.
F2: Learning curves. Training patients vary; tuning and calibration labels are fixed. Bars are
SD over three trained seeds, not confidence intervals. Main text: select Brier and AUROC panels;
NLL and ECE panels can be supplementary. No plateau/non-inferiority inference is automated.
F3: Reliability diagrams (15 equal-width probability bins). Seed 17 is prespecified for individual
models; the three-seed ensemble is explicitly labeled. Empty bins are omitted. Bin counts are
saved as CSV. Sparse high-probability bins should not be overinterpreted.
F4: Primary paired Brier-score changes, temperature minus raw. Negative values favor
source-fitted temperature scaling. Intervals resample test patients (images for VinBigData),
conditional on fitted models/calibrators. Multiplicity-adjusted p values are in Table T3.
F5: Selective error and positive-case coverage using entropy. Decision thresholds were selected
in source calibration to target 90% sensitivity and then frozen. These are retrospective
reference-label assessments, not a validated clinical referral policy.
F6: Calibration-label-budget sensitivity at 100% training. Failed low-event fits are tabulated,
not replaced with an identity calibrator or treated as successful.
SF1: ROC and PR curves for calibrated three-seed ensembles. PR curves depend on cohort prevalence.

All figure panels are exported individually as 300-dpi PNG, vector PDF and editable SVG.
Choose a readable subset for the main manuscript; retain remaining panels in a supplement.
