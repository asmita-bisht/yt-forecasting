"""
test_threshold.py

This test validates that the probability to label thresholding behaves monotonically.

"""
import numpy as np
import pytest
import pandas as pd

def test_threshold_outputs_binary(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found")
    from tests.conftest import apply_threshold
    p = model.predict_proba(tiny_raw_df)[:, -1]
    y = apply_threshold(p, 0.5)
    assert set(y).issubset({0,1})

@pytest.mark.parametrize("thr_low,thr_high", [(0.2,0.3),(0.5,0.6),(0.8,0.9)])
def test_higher_threshold_never_increases_positives(model, tiny_raw_df, thr_low, thr_high):
    if model is None:
        pytest.skip("Model not found")
    from tests.conftest import apply_threshold
    p = model.predict_proba(tiny_raw_df)[:, -1]
    y_low  = apply_threshold(p, thr_low)
    y_high = apply_threshold(p, thr_high)
    assert y_high.sum() <= y_low.sum(), "Raising threshold increased positives"
