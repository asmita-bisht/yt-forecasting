"""
test_model_roundtrip.py

This test verifies model serialization stability.

"""

import os, tempfile
import numpy as np
import pytest
import joblib
import pandas as pd

def _predict_proba(model, df: pd.DataFrame):
    # Your model is a Pipeline that expects RAW columns.
    return model.predict_proba(df)[:, -1]

def test_predict_proba_roundtrip(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found in models/*.joblib")
    p1 = _predict_proba(model, tiny_raw_df)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pkl")
        joblib.dump(model, path)
        m2 = joblib.load(path)

    p2 = _predict_proba(m2, tiny_raw_df)
    assert np.allclose(p1, p2, atol=1e-10), "Predictions changed after serialize→deserialize"
