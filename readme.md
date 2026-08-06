# 🔧 Predictive Maintenance using LSTM

Predicts the **Remaining Useful Life (RUL)** of aircraft engines from sensor telemetry, using an LSTM trained on NASA's C-MAPSS (FD001) turbofan degradation dataset. Includes a Streamlit app for interactive predictions on uploaded engine data.

## Demo

Upload a CSV of an engine's recent sensor readings and get back a predicted number of cycles remaining before maintenance is needed.

<!-- Add a screenshot here, e.g.: -->
<!-- ![App screenshot](assets/app_screenshot.png) -->

## How it works

1. **Data**: NASA C-MAPSS FD001 — simulated turbofan engines run to failure under a single operating condition, with 21 sensor channels + 3 operational settings recorded per cycle.
2. **Preprocessing**:
   - Drop 7 sensors with near-zero variance (uninformative for RUL).
   - Compute RUL per cycle as `max_cycle - current_cycle`, capped at 125 (standard for this dataset — early-life RUL is too noisy to fit reliably otherwise).
   - Scale all 17 remaining features (`op_setting_1/2/3` + 14 sensors) with `StandardScaler`, fit on train only.
3. **Sequencing**: Build sliding windows of 30 consecutive cycles per engine as model input; test sequences shorter than 30 cycles are zero-padded.
4. **Model**: 2-layer LSTM (128 → 64 units) with dropout, followed by dense layers down to a single RUL output. Trained with Huber loss, early stopping on validation loss, and a train/validation split done **by engine** (not by row) to avoid data leakage between splits.
5. **Deployment**: A Streamlit app loads the trained model, scaler, and feature list, and scores newly uploaded engine data through the same pipeline used at training time.

## Results

| Metric | Value |
|---|---|
| MAE  | 11.34
| RMSE | 16.31

## Project structure

```
.
├── app.py                       # Streamlit inference app
├── rul_lstm_fixed.py            # Training pipeline (data → model → saved artifacts)
├── lstm_rul_model.keras         # Trained model (full architecture + weights)
├── lstm_rul_model.weights.h5    # Trained weights only (used by app.py for version-safe loading)
├── scaler.pkl                   # StandardScaler fit on training data
├── feature_columns.pkl          # Exact feature list/order used at training time
├── requirements.txt
├── train_FD001.txt … train_FD004.txt   # C-MAPSS training trajectories
├── test_FD001.txt … test_FD004.txt     # C-MAPSS test trajectories
├── RUL_FD001.txt … RUL_FD004.txt       # Ground-truth RUL for test engines
└── README.md
```

> This repo currently trains/evaluates on **FD001** (single operating condition, single fault mode). FD002–FD004 are included for future extension but aren't yet used by the training script.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Training

Retrain from scratch (regenerates `lstm_rul_model.keras`, `lstm_rul_model.weights.h5`, `scaler.pkl`, and `feature_columns.pkl` together, so they stay consistent):

```bash
python rul_lstm_fixed.py
```

## Running the app locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload a CSV containing the 17 expected columns (`op_setting_1`, `op_setting_2`, `op_setting_3`, and 14 sensor columns — see `feature_columns.pkl` or the app's error message for the exact list), and click **Predict Remaining Useful Life**.

## Notes on the pipeline

- The model, scaler, and feature list **must all come from the same training run**. Mixing an old scaler with a newer model (or vice versa) causes silent or hard-to-debug prediction errors — always regenerate all four artifact files together via `rul_lstm_fixed.py`, never individually.
- RUL predictions are capped at 125 cycles by design, matching how the model was trained — this isn't a bug if predictions never exceed that ceiling.

## Dataset

[NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). Check NASA's data usage terms before redistributing the raw files if you fork this into a public repo.

