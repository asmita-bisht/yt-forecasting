"""
test_reproducibility.py

This test ensures repeatable predictions when seeds are fixed.

"""
import numpy as np
import random
import pytest

def _set_all_seeds(seed=13):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_prediction_reproducible(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found")
    _set_all_seeds(42)
    p1 = model.predict_proba(tiny_raw_df)[:, -1]
    _set_all_seeds(42)
    p2 = model.predict_proba(tiny_raw_df)[:, -1]
    assert np.allclose(p1, p2, atol=1e-12)
