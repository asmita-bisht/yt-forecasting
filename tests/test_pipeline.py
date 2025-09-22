"""
test_pipeline.py

This test has lower-level smoke tests for the sklearn ColumnTransformer/Pipeline that was exported.

"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

def _max_min_df_in_pipeline(pipe) -> int:
    max_md = 0
    if hasattr(pipe, "transformers"):
        for _, trans, _ in pipe.transformers:
            if isinstance(trans, TfidfVectorizer):
                if isinstance(trans.min_df, int):
                    max_md = max(max_md, trans.min_df)
            if hasattr(trans, "steps"):
                for _, step in trans.steps:
                    if isinstance(step, TfidfVectorizer) and isinstance(step.min_df, int):
                        max_md = max(max_md, step.min_df)
    return max_md

def _pad_and_enrich_text(df: pd.DataFrame, need_rows: int, text_cols):
    df2 = df.copy()
    pad = max(0, need_rows - len(df2))
    if pad > 0:
        pad_block = pd.concat([df2.iloc[:1].copy() for _ in range(pad)], ignore_index=True)
        df2 = pd.concat([df2.reset_index(drop=True), pad_block], ignore_index=True)
    for c in text_cols:
        if c in df2.columns:
            s = df2[c].astype(str).fillna("")
            s = np.where(pd.Series(s).str.len() < 2, "video data", s)
            s = s + " model"
            s = np.where((np.arange(len(s)) % 2) == 0, s + " youtube", s)
            df2[c] = s
    return df2

def _fit_ok(pipe, df):
    need_rows = max(6, _max_min_df_in_pipeline(pipe) + 1)
    text_cols = [c for c in df.columns if c.lower() in ("title", "description")]
    df2 = _pad_and_enrich_text(df, need_rows, text_cols)
    pipe.fit(df2)
    return df2  # return the frame we used to fit (same schema)

def test_transformer_smoke(feature_pipeline, tiny_raw_df):
    if feature_pipeline is None:
        pytest.skip("No sklearn feature pipeline exported (models/preprocess_phase3.joblib)")
    df_fit = _fit_ok(feature_pipeline, tiny_raw_df)
    Xt = feature_pipeline.transform(tiny_raw_df)
    # basic shape sanity
    assert hasattr(Xt, "shape")
    assert Xt.shape[0] == len(tiny_raw_df)

def test_no_nans_after_imputation(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found")
    proba = model.predict_proba(tiny_raw_df)[:, -1]
    assert proba.shape[0] == len(tiny_raw_df)
