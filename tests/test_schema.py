"""
test_schema.py

This test guards the raw input schema the model expects.

"""
import pandas as pd

def test_required_columns_present(tiny_raw_df, input_schema):
    req = set(input_schema.keys())
    assert req.issubset(tiny_raw_df.columns)

def test_dtypes_strict(tiny_raw_df, input_schema):
    for col, expected in input_schema.items():
        if expected == "category":
            assert pd.api.types.is_categorical_dtype(tiny_raw_df[col])
        else:
            assert str(tiny_raw_df[col].dtype) == expected

def test_basic_value_ranges(tiny_raw_df):
    num_cols = [c for c in tiny_raw_df.columns if pd.api.types.is_numeric_dtype(tiny_raw_df[c])]
    for c in num_cols:
        assert tiny_raw_df[c].isna().mean() == 0.0

def test_no_duplicate_columns(tiny_raw_df):
    assert tiny_raw_df.columns.is_unique
