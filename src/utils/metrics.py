"""评估指标：Macro-F1（任务书 4.3 节规定）"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom > 0 else 0.0
    return f1s


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    return float(per_class_f1(y_true, y_pred, num_classes).mean())


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
