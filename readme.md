# Predictive Maintenance using LSTM

An LSTM-based deep learning system for predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor time-series data.

## Overview

This project explores Remaining Useful Life prediction as a time-series regression problem. The model learns temporal degradation patterns from historical engine sensor measurements and predicts how many operational cycles remain before failure.

The project uses the **NASA C-MAPSS FD001** dataset and an LSTM-based sequence model.

## Approach

The pipeline consists of:

1. Data loading and preprocessing
2. Sensor/feature selection
3. RUL calculation and capping
4. Feature standardization
5. Sliding-window sequence construction
6. Engine-level train/validation splitting
7. LSTM model training
8. Evaluation on held-out data
9. Interactive inference through Streamlit

## Dataset

The experiments use **NASA C-MAPSS FD001**, a turbofan engine degradation dataset containing multivariate sensor measurements collected over engine operating cycles.

The project uses the available operating-condition and sensor measurements to construct temporal sequences for RUL prediction.

## Temporal Modeling

The model uses **30-cycle sliding windows**.

Each training example therefore contains a sequence of observations rather than a single independent row. This allows the LSTM to learn temporal patterns associated with degradation.

## Model

The core model is a two-layer LSTM:

* LSTM: 128 units
* Dropout
* LSTM: 64 units
* Output layer for RUL regression

Training uses **Huber loss** and early stopping.

## Evaluation

On the current test evaluation:

| Metric |    Result |
| ------ | --------: |
| MAE    | **11.34** |
| RMSE   | **16.31** |

These metrics measure the difference between predicted and actual Remaining Useful Life.

## Validation Strategy

The project uses an **engine-level split** rather than randomly splitting individual time-series rows.

This is important because observations from the same engine are temporally related. Splitting individual rows across training and validation sets could allow information from the same degradation trajectory to appear in both sets.

## Application

A Streamlit interface is included for interacting with the trained model and visualizing predicted versus actual RUL.

## Limitations

The current experiment should be interpreted as a controlled research/learning project rather than a production predictive-maintenance system.

Important limitations include:

* Evaluation is currently focused on the FD001 subset.
* The model has not been demonstrated across multiple operating regimes.
* The current evaluation does not establish robustness to distribution shift.
* RUL prediction performance may change substantially under different preprocessing, sequence lengths, or engine populations.
* Further experiments are needed before making claims about real-world deployment.

## Future Work

Potential extensions include:

* Evaluate across additional C-MAPSS subsets.
* Compare LSTM performance with GRU, temporal convolutional networks, and Transformer-based approaches.
* Investigate how sequence length affects RUL prediction.
* Perform systematic hyperparameter experiments.
* Analyze model behavior under distribution shift.
* Add uncertainty estimation to RUL predictions.
* Investigate whether learned temporal representations remain stable when operating conditions change.

## Project Structure

```text
Predictive-Maintenance/
│
├── data/
├── notebooks/
├── models/
├── app/
├── requirements.txt
├── README.md
└── ...
```

## Tech Stack

**Python · TensorFlow/Keras · LSTM · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit**

## Author

**Lakshya Chaudhary**

B.Tech in Electrical Engineering
National Institute of Technology, Patna
