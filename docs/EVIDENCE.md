# Evidence and result provenance

[← Portfolio](../README.md)

Published October 26, 2026. Primary rank evidence is the user-supplied screenshot named `Screenshot 2026-10-25 222252.png`, preserved without edits as [`assets/kaggle-standings-2026-10-25.png`](../assets/kaggle-standings-2026-10-25.png). Its filename dates the snapshot to October 25, 2026; an independent screenshot capture timestamp was not checked.

| Claim | Evidence | Scope |
|---|---|---|
| Petals rank 12/50 | [Supplied dashboard screenshot](../assets/kaggle-standings-2026-10-25.png) | Ongoing leaderboard snapshot |
| Connect X rank 41/255 | [Same dashboard screenshot](../assets/kaggle-standings-2026-10-25.png) | Ongoing leaderboard snapshot; supersedes earlier local rank records |
| Flower Kaggle macro F1 0.94693 | [Accepted-submission screenshot](../assets/flower-submission-scores.png), [result record](../results/achievements.json) | Saved ensemble version 362195726 |
| V2S validation macro F1 0.94826999 | [Validation report](../projects/petals-to-the-metal/reports/v2s/validation_report.json) | Official held-out split, four-view averaging |
| B3 validation macro F1 0.94558873 | [Validation report](../projects/petals-to-the-metal/reports/b3/validation_report.json) | Official held-out split, four-view averaging |
| Ensemble validation macro F1 0.95479064 | [Blend report](../projects/petals-to-the-metal/reports/ensemble_report.json) | Validation metric used to select blend weights |
| Calibration rejected | [Calibration report](../projects/petals-to-the-metal/reports/calibration_result.json) | Not submitted |
| Native timing, depth, nodes, 308 board checks, baseline games | [Runtime report](../projects/connect-x/reports/runtime_validation.json), [screenshot](../assets/connectx-runtime-validation.png) | Original observed Kaggle CPU test; no guarantee for other positions |

The flower reports and Connect X runtime report were transcribed from observed saved Kaggle notebook output during development. The Connect X report contains a contemporaneous note that competition validation was pending; accepted-submission evidence recorded later supersedes that earlier status. Its measurements remain historical test evidence.

The source and JSON records permit inspection but are not a third-party audit. Saved Kaggle notebook links in each case study provide additional provenance, subject to notebook visibility. Source code in this repository corresponds to the scored flower pipeline and accepted native game agent. Later full-data flower refit experiments are outside this snapshot and are not represented as a scored improvement.

Competition links: [Petals leaderboard](https://www.kaggle.com/competitions/tpu-getting-started/leaderboard) · [Connect X leaderboard](https://www.kaggle.com/competitions/connectx/leaderboard). Live ranks may differ from the dated screenshot.
