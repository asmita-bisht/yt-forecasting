"""
test_explainability_smoke.py

This test is a lightweight explainability SHAP smoke test. 

"""
import pytest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*LGBMClassifier")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.pipeline")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shap\.explainers\._kernel")


def _augment_text_for_tfidf(df: pd.DataFrame, text_cols, min_rows: int = 6):
    """
    Make sure TF-IDF with min_df and stopwords can build a vocab.
    Pads to `min_rows` using the first row, then inserts non-stopword tokens.
    """
    df2 = df.copy()

    # --- pad to at least `min_rows` rows ---
    pad = max(0, min_rows - len(df2))
    if pad > 0:
        pad_block = pd.concat([df2.iloc[:1].copy() for _ in range(pad)], ignore_index=True)
        df2 = pd.concat([df2.reset_index(drop=True), pad_block], ignore_index=True)

    # --- add simple tokens to text columns to satisfy min_df ---
    for c in text_cols:
        if c in df2.columns:
            s = df2[c].astype(str).fillna("")
            # avoid pure stopwords
            s = np.where(pd.Series(s).str.len() < 2, "video data", s)
            # repeat some non-stop tokens to hit min_df across rows
            s = s + " model"
            s = np.where((np.arange(len(s)) % 2) == 0, s + " youtube", s)
            df2[c] = s
    return df2

def test_shap_smoke(model, tiny_raw_df):
    if model is None:
        pytest.skip("Model not found")
    try:
        import shap
    except Exception:
        pytest.skip("SHAP not installed")

    # Use RAW columns (the pipeline inside `model` will do its own transforms).
    cols = list(tiny_raw_df.columns)

    # Heuristically guess your text columns (common in your project)
    guess_text = [c for c in cols if c.lower() in ("title", "description")]
    df_raw = tiny_raw_df.copy()
    df_raw = _augment_text_for_tfidf(df_raw, guess_text)

    # Build a small background + test set
    Xb = df_raw.sample(min(20, len(df_raw)), random_state=0)
    bg  = Xb.iloc[:10, :].copy()
    tst = Xb.iloc[10:, :].copy()
    if len(tst) == 0:  # if tiny frame
        tst = bg.iloc[:1, :].copy()

    # KernelExplainer sends numpy arrays to f(X); wrap back to a DataFrame with the right columns
    def f(x_np: np.ndarray):
        x_df = pd.DataFrame(x_np, columns=cols).astype(df_raw.dtypes.to_dict(), errors="ignore")
        return model.predict_proba(x_df)[:, -1]

    explainer = shap.KernelExplainer(f, bg.to_numpy())
    vals = explainer.shap_values(tst.to_numpy(), nsamples=50)
    if isinstance(vals, list):  # multiclass fallback
        vals = vals[-1]
    assert len(vals) == len(tst)
