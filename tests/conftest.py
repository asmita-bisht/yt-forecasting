"""
conftest.py

This file provides reusable fixtures and utilities for all of the tests.

"""


import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys, importlib
from sklearn.exceptions import NotFittedError

# --- repo roots based on your tree ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

def _latest(globpat: str):
    hits = glob.glob(globpat)
    return max(hits, key=os.path.getmtime) if hits else None

# --------- Artifacts: pipeline & model loaders ----------
def build_feature_pipeline():
    """
    If you exported a preprocessing pipeline, load it.
    Expected path (from your tree): models/preprocess_phase3.joblib
    """
    path = MODELS_DIR / "preprocess_phase3.joblib"
    if path.exists():
        import joblib
        return joblib.load(path)
    return None  # no sklearn pipeline exported

def load_model():
    """
    Load the newest ensemble model from models/.
    Prefer names containing 'Ensemble' and '.joblib', skip the preprocess artifact.
    """
    # prefer ensemble-named models
    prefer = sorted(
        MODELS_DIR.rglob("*Ensemble*.joblib"),
        key=lambda p: p.stat().st_mtime,
    )
    if prefer:
        import joblib
        return joblib.load(prefer[-1])

    # otherwise take the newest .joblib that is not the preprocess
    candid = sorted(
        [p for p in MODELS_DIR.rglob("*.joblib") if p.name != "preprocess_phase3.joblib"],
        key=lambda p: p.stat().st_mtime,
    )
    if candid:
        import joblib
        return joblib.load(candid[-1])

    return None  # tests that need a model will skip

# --------- Data discovery (for schema + tiny fixtures) ----------
def _load_latest_ml_table():
    """
    Load a small sample from the newest processed feature table:
    data/processed/ml_table_*.parquet  (from your tree)
    Fallback to phase2_latest.parquet if needed.
    """
    # newest ml_table_*.parquet
    latest_ml = _latest(str(DATA_PROCESSED / "ml_table_*.parquet"))
    if latest_ml is None:
        alt = DATA_PROCESSED / "phase2_latest.parquet"
        latest_ml = str(alt) if alt.exists() else None

    if latest_ml is None:
        return None

    df = pd.read_parquet(latest_ml) if latest_ml.endswith(".parquet") else pd.read_csv(latest_ml)
    # Drop likely targets if present
    for tgt in ("viral", "label", "y", "target"):
        if tgt in df.columns:
            df = df.drop(columns=[tgt])
    return df

def _schema_from_df(df: pd.DataFrame):
    # Convert pandas dtypes to simple strings for assertions
    def _dtstr(dt):
        s = str(dt)
        # normalize "category" check
        if "category" in s:
            return "category"
        return s
    return {c: _dtstr(dt) for c, dt in df.dtypes.items()}

# --------- Pytest fixtures ----------
@pytest.fixture(scope="session")
def feature_pipeline():
    return build_feature_pipeline()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def model():
    """
    Ensure SoftVotingEnsemble is importable under the same module path used at training time,
    THEN load the newest ensemble artifact.
    """
    # 1) Preload and alias the class/module so pickle can resolve it
    try:
        # import the module so 'models.soft_voting_ensemble' exists
        sve_mod = importlib.import_module("models.soft_voting_ensemble")
        # expose SoftVotingEnsemble under __main__ too (covers pickles saved from __main__)
        import types
        if "__main__" not in sys.modules:
            sys.modules["__main__"] = types.ModuleType("__main__")
        sys.modules["__main__"].SoftVotingEnsemble = getattr(sve_mod, "SoftVotingEnsemble")
    except Exception as e:
        # If import fails, we'll still attempt to load; if that fails we skip tests downstream.
        pass

    # 2) Find newest model artifact and load
    import joblib
    MODELS_DIR = ROOT / "models"
    prefer = sorted(MODELS_DIR.rglob("*Ensemble*.joblib"), key=lambda p: p.stat().st_mtime)
    candidates = prefer or sorted(
        [p for p in MODELS_DIR.rglob("*.joblib") if p.name != "preprocess_phase3.joblib"],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        pytest.skip("Model not found under models/**/*.joblib")

    try:
        return joblib.load(candidates[-1])
    except ModuleNotFoundError as e:
        # Last-resort helpful message for debugging import path issues
        print(f"[conftest] Could not import dependency while unpickling: {e}")
        return None

@pytest.fixture(scope="session")
def input_schema():
    """
    In Layer-1 tests we validate 'inference-time input' schema.
    We infer it from your newest processed table (features already built).
    If you prefer to assert a *raw* schema, replace this with a dict you control.
    """
    df = _load_latest_ml_table()
    if df is not None and len(df) > 0:
        # keep a reasonable number of columns if extremely wide
        if df.shape[1] > 500:
            df = df.iloc[:, :500]
        return _schema_from_df(df)
    # Fallback minimal schema if no file found
    return {
        "title": "object",
        "views": "int64",
        "likes": "int64",
        "duration": "float64",
    }

@pytest.fixture()
def tiny_raw_df(input_schema):
    """
    Make a tiny, realistic frame that matches the discovered schema.
    Since your app predicts on engineered features (processed ml_table),
    this fixture represents 'inference-time features'.
    """
    # Build 3 rows with dtype-aware defaults
    data = {}
    for col, dt in list(input_schema.items())[:200]:   # cap width for sanity
        if dt.startswith("int"):
            data[col] = [0, 1, 10]
        elif dt.startswith("float"):
            data[col] = [0.0, 1.5, 10.0]
        elif dt in ("category", "object", "string"):
            data[col] = ["a", "b", ""]  # short strings; category cast later if needed
        elif dt.startswith("bool"):
            data[col] = [False, True, False]
        else:
            # default to object string
            data[col] = ["a", "b", ""]
    df = pd.DataFrame(data)

    # Enforce categories if any were detected
    for col, dt in input_schema.items():
        if dt == "category":
            df[col] = df[col].astype("category")
        else:
            # attempt strict cast for numeric-like
            try:
                df[col] = df[col].astype(dt)
            except Exception:
                pass  # leave as-is if cast fails (some pandas dtype strings vary)
    return df

def _augment_text_df(df: pd.DataFrame, text_cols):
    """
    Ensure Tfidf(min_df=k) can build a vocab: add a few simple, repeated tokens
    across rows so every text col has >= min_df occurrences of something non-stopword.
    """
    if df.shape[0] < 5:
        # pad to 5 rows to comfortably satisfy typical min_df (e.g., 3)
        pad = 5 - df.shape[0]
        df = pd.concat([df, df.iloc[:1].copy().repeat(pad)]).reset_index(drop=True)
    for c in text_cols:
        if c in df.columns:
            # Put repeated non-stopwords; avoid pure stopwords like "a", "the"
            df[c] = df[c].fillna("")
            mask_emptyish = df[c].str.len().lt(2)
            df.loc[mask_emptyish, c] = "video data"
            # add some repetition to hit min_df
            df.loc[df.index % 2 == 0, c] = df[c] + " model"
            df.loc[df.index % 3 == 0, c] = df[c] + " youtube"
    return df

@pytest.fixture()
def X_small(tiny_raw_df, feature_pipeline):
    """
    Try transform-only (if pipeline is already fitted). If not fitted or the
    Tfidf block complains about empty vocabulary, augment text minimally and fit.
    Return a pandas DataFrame either way.
    """
    # No exported pipeline → identity
    if feature_pipeline is None:
        return tiny_raw_df.copy()

    # Try transform-only first (best case: pipeline was saved *fitted*)
    Xt = None
    try:
        Xt = feature_pipeline.transform(tiny_raw_df)
    except (NotFittedError, AttributeError, ValueError):
        # Need to fit: augment text columns so Tfidf(min_df) can build vocab
        text_cols = []
        try:
            # Heuristic: sniff vectorizer columns from transformers
            for name, trans, cols in getattr(feature_pipeline, "transformers", []):
                if hasattr(trans, "get_params") and "stop_words" in trans.get_params():
                    # Single vectorizer directly
                    if isinstance(cols, (list, tuple)):
                        text_cols.extend(list(cols))
                    else:
                        text_cols.append(cols)
                # Pipelines inside ColumnTransformer
                if hasattr(trans, "steps"):
                    for step_name, step in trans.steps:
                        if hasattr(step, "get_params") and "stop_words" in step.get_params():
                            if isinstance(cols, (list, tuple)):
                                text_cols.extend(list(cols))
                            else:
                                text_cols.append(cols)
        except Exception:
            pass
        text_cols = list(dict.fromkeys([c for c in text_cols if isinstance(c, str)]))  # unique & valid
        df_fit = _augment_text_df(tiny_raw_df.copy(), text_cols)
        Xt = feature_pipeline.fit(df_fit).transform(df_fit.iloc[: len(tiny_raw_df)])

    # Wrap numpy/sparse output back to DataFrame if possible
    cols = None
    try:
        cols = feature_pipeline.get_feature_names_out()
    except Exception:
        pass
    if hasattr(Xt, "toarray"):  # sparse
        Xt = Xt.toarray()
    if isinstance(Xt, np.ndarray):
        if cols is None:
            cols = [f"f{i}" for i in range(Xt.shape[1])]
        Xt = pd.DataFrame(Xt, columns=list(cols))
    return Xt

def apply_threshold(p, thr: float):
    """Tiny helper used by tests to turn probs -> labels."""
    return (np.asarray(p) >= float(thr)).astype(int)
