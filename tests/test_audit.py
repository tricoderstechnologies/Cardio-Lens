import numpy as np
from src.audit import INSUFFICIENT, age_group, subgroup_row


def test_small_group_flagged_and_no_auc():
    r = subgroup_row("x", "g", [0, 1] * 5, [0, 1] * 5, np.linspace(0, 1, 10))
    assert r["roc_auc"] is None and r["note"] == INSUFFICIENT and not r["reliable"]


def test_auc_reported_when_thresholds_met():
    y = [0] * 15 + [1] * 15
    r = subgroup_row("x", "g", y, y, np.array(y) * 0.8 + 0.1)
    assert r["reliable"] and r["roc_auc"] == 1.0 and r["note"] == ""


def test_rates_are_correct():
    r = subgroup_row("x", "g", [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0], [.9, .8, .2, .1, .7, .1])
    assert r["recall"] == 0.5 and r["false_negative_rate"] == 0.5 and r["false_positive_rate"] == 0.5


def test_age_bins_boundaries():
    import pandas as pd
    g = age_group(pd.Series([44, 45, 59, 60]))
    assert list(g.astype(str)) == ["Below 45", "45-59", "45-59", "60 and above"]
