# PSD·주기성 Ablation 결과

A/B inference and test evaluation completed with the fixed signed checkpoint. C/D fusion is unavailable under the specified one-sided-score plus normal-train IQR normalization: both feature-score IQRs are zero. No epsilon or alternate normalization was substituted.

```json
[
  {
    "method": "A_AE",
    "available": true,
    "threshold": 0.0013263815781101584,
    "precision": 0.8924731182795699,
    "recall": 0.9325842696629213,
    "f1": 0.9120879120879121,
    "false_positive_rate": 0.004987531172069825,
    "false_negatives": 6,
    "confusion_matrix": [
      [
        1995,
        10
      ],
      [
        6,
        83
      ]
    ],
    "pr_auc": 0.9700489981053247
  },
  {
    "method": "B_PSD",
    "available": true,
    "threshold": 0.32678645610699497,
    "precision": 0.5329341317365269,
    "recall": 1.0,
    "f1": 0.6953124999999999,
    "false_positive_rate": 0.03890274314214464,
    "false_negatives": 0,
    "confusion_matrix": [
      [
        1927,
        78
      ],
      [
        0,
        89
      ]
    ],
    "pr_auc": 1.0
  },
  {
    "method": "C_AE_PSD",
    "available": false,
    "reason": "psd_score: normal-train IQR is zero/too small (0.0); score intentionally not fabricated"
  },
  {
    "method": "D_AE_ACF",
    "available": false,
    "reason": "acf_score: normal-train IQR is zero/too small (0.0); score intentionally not fabricated"
  }
]
```
