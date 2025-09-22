# Forecasting Video Performance on YouTube
*A data-driven machine learning project on engagement signals*

This project develops and evaluates a predictive machine learning model that estimates the likelihood a YouTube video will rank in the top performance tier compared to similar uploads. In addition to exploratory data analysis (EDA) to uncover engagement patterns, the workflow builds and validates machine learning models using pre-publish metadata (title, description, tags, category, and publish timing). A companion web app built using Streamlit showcases the model in action, allowing users to input video metadata and receive an interactive forecast with explanatory insights, demonstrating the potential of predictive modeling for content strategy

This project was completed as the Final Project for CM3070 (Final Project in Computer Science), part of the BSc Computer Science program at the University of London, Goldsmiths.

## Quickstart

```bash
# 0) Clone & enter
git clone yt-forecast.git
cd yt-forecasting

# 1) Create & activate a virtual env (pick one)
python -m venv .venv && source .venv/bin/activate
or: conda create -n yt-forecast python=3.10 -y && conda activate yt-forecast

# 2) Install dependencies (pyproject preferred)
pip install -e .[dev]
pip install -r requirements.txt

# 3) Configure secrets
cp .env.example .env
# open .env and set YOUTUBE_API_KEY (and any other required values)

# 4) Run the pipeline (Reccomended: instead of make commands, download repo and run the notebooks interactively for yourself to see how it works!)
make data       # Phase 1: collect raw → interim
make dedupe     # deduplicate interim → processed
make train      # Phase 3: model training + evaluation
make analysis   # Phase 4: reporting & insights

# 5) Launch the demo app
make app
# or directly:
streamlit run app.py

```

## Notebooks

Each notebook starts with a short header and links back to this README for setup. Suggested order:

1. `Phase1_Data_Collection.ipynb` – explore raw data, schema, data quality
2. `Phase2_Cleaning_and_Feature_Engineering.ipynb` – clean & feature table, labeling
3. `Phase3_Modeling_Eval_Calibration.ipynb` – train baselines/tuned models
4. `Phase4_Analysis.ipynb` – metrics, curves, explanations

> Keep notebooks lightweight; move reusable code into `scripts/`.

## Project layout

```
yt-forecasting/
├── app.py                      # Streamlit demo app
├── Makefile                    # One-command pipeline runner
├── pyproject.toml              # Project metadata + dependencies (preferred)
├── requirements.txt            # (Fallback) pinned deps
├── pytest.ini                  # Pytest config
├── README.md
│
├── notebooks/                  # Reproducible research / reports
│   ├── Phase1_Data_Collection.ipynb
│   ├── Phase2_Cleaning_and_Features.ipynb
│   ├── Phase3_Modeling_Eval_Calib.ipynb
│   └── Phase4_Analysis.ipynb
│
├── scripts/                    
│   └── combine_and_dedupe.py   # de-duplication / merging step
│
├── models/
│   ├── soft_voting_ensemble.py # custom estimator used by the app/bundle
│   └── bundles/                # (ignored) trained model artifacts (.joblib)
│
├── data/                       # (ignored) local datasets
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── reports/
│   └── executed/               # (ignored) executed notebooks, metrics, plots
│
└── tests/                      # Automated checks
    ├── conftest.py
    ├── test_app_e2e.py
    ├── test_explainability_smoke.py
    ├── test_features.py
    ├── test_model_roundtrip.py
    ├── test_pipeline.py
    ├── test_reproducibility.py
    ├── test_schema.py
    └── test_threshold_logic.py

```
## Reproducibility & tracking
- Set seeds; record versions in `models/<run>/meta.json` and `reports/metrics.json`.
- Save the **model + preprocessing pipeline** together.
- Track experiments (MLflow optional); at minimum commit metrics JSON.

## Environment variables

Create `.env` from `.env.example`:

```env
YOUTUBE_API_KEY=your_key_here
```

## Troubleshooting

- **HTTP 403 / quota**: throttle `SEARCH_UNITS` or wait for quota reset.
- **`ModuleNotFoundError`** in notebooks: run `pip install -e .[dev]` and restart kernel.
- **Pickle load errors**: keep custom estimators importable at `src/models/...`.
>>>>>>> 7771240 (Initial commit)
