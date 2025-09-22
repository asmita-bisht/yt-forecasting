"""
test_adversarial_inputs.py

This tests basic robustness against adversial or weird inputs. 

"""

import numpy as np
import pandas as pd
import pytest

def test_adversarial_inputs_do_not_crash(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found")

    df = tiny_raw_df.copy()

    # push some numeric columns to extremes + NaN
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols[:3]:
        df.loc[0, c] = 1e12
        df.loc[1, c] = -1e6
        df.loc[2, c] = np.nan

    # make one text column extremely long
    for c in df.columns:
        if df[c].dtype == "object":
            df.loc[2, c] = "🙂" * 2000
            break

    # call your full model pipeline on RAW df
    proba = model.predict_proba(df)[:, -1]
    assert np.isfinite(proba).all(), "Non-finite probabilities for adversarial inputs"
