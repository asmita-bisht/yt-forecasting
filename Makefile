# ===== Notebook-driven pipeline (yt-forecasting) =====
NB_SRC      := notebooks
NB_OUT      := reports/executed
DATA_RAW    := data/raw
DATA_INTER  := data/interim
DATA_PROC   := data/processed
MODELS_DIR  := models
REPORTS_DIR := reports

# Wildcards so minor filename tweaks still work
NB_PHASE1   := $(NB_SRC)/Phase1_*.ipynb
NB_PHASE2   := $(NB_SRC)/Phase2_*.ipynb
NB_PHASE3   := $(NB_SRC)/Phase3_*.ipynb
NB_PHASE4   := $(NB_SRC)/Phase4_*.ipynb

.PHONY: all data dedupe features train eval analysis app clean

# Full pipeline in your sequence
all: data dedupe train analysis

# 1) Phase 1 — Data Collection
data:
	mkdir -p $(NB_OUT) $(DATA_RAW) $(DATA_INTER)
	jupyter nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=1800 \
	  --output $(NB_OUT)/Phase1_Data_Collection.out.ipynb \
	  $(NB_PHASE1)

# (Optional) Phase 2 — Cleaning & Feature Engineering (if you want to run it as a notebook step)
features:
	mkdir -p $(NB_OUT) $(DATA_PROC)
	jupyter nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=2400 \
	  --output $(NB_OUT)/Phase2_Cleaning_and_Feature_Engineering.out.ipynb \
	  $(NB_PHASE2)

# 2) Dedupe script (between Phase 1 and Phase 3, as you requested)
# Adjust script path/flags if different.
dedupe:
	mkdir -p $(DATA_PROC)
	python scripts/combine_and_dedupe.py \
	  --in_dir $(DATA_INTER) \
	  --out_dir $(DATA_PROC)

# 3) Phase 3 — Modeling / Eval / Calibration
train:
	mkdir -p $(NB_OUT) $(MODELS_DIR) $(REPORTS_DIR)
	jupyter nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=3600 \
	  --output $(NB_OUT)/Phase3_Modeling_Eval_Calibration.out.ipynb \
	  $(NB_PHASE3)

# Alias, since Phase 3 already evaluates
eval: train

# 4) Phase 4 — Analysis / Reporting
analysis:
	mkdir -p $(NB_OUT) $(REPORTS_DIR)
	jupyter nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=1800 \
	  --output $(NB_OUT)/Phase4_Analysis.out.ipynb \
	  $(NB_PHASE4)

# Launch Streamlit demo
app:
	streamlit run app.py

clean:
	rm -rf $(NB_OUT)
