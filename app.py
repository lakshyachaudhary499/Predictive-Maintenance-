import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance using LSTM")
st.write("Predict the Remaining Useful Life (RUL) of an aircraft engine.")

SEQUENCE_LENGTH = 30


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    n_features = len(feature_columns)

    # Rebuild the exact architecture used at training time, then load
    # only the numeric weights (.weights.h5). This avoids the
    # full-model .keras deserialization path, which breaks across
    # Keras minor-version differences (e.g. GlorotUniform config
    # gaining new fields like input_axes/output_axes).
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, n_features)),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.load_weights("lstm_rul_model.weights.h5")

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()


def pad_sequence(seq, seq_length, n_feats):
    if len(seq) >= seq_length:
        return seq[-seq_length:]
    pad = np.zeros((seq_length - len(seq), n_feats))
    return np.vstack([pad, seq])


uploaded_file = st.file_uploader("Upload Engine Data (.csv)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    with st.expander("Preview uploaded data"):
        st.dataframe(data.head())
        st.write(f"Rows: {len(data)}  |  Columns: {len(data.columns)}")

    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        st.error(
            "Uploaded CSV is missing columns the model needs: "
            f"{missing}. Expected columns: {feature_columns}"
        )
        st.stop()

    if "unit_number" in data.columns and data["unit_number"].nunique() > 1:
        unit = st.selectbox("Select engine (unit_number)", sorted(data["unit_number"].unique()))
        engine_data = data[data["unit_number"] == unit]
    else:
        engine_data = data

    if st.button("Predict Remaining Useful Life"):
        try:
            features = engine_data[feature_columns]
            scaled = scaler.transform(features)
            sequence = pad_sequence(scaled, SEQUENCE_LENGTH, len(feature_columns))
            sequence = np.expand_dims(sequence, axis=0)

            prediction = model.predict(sequence)
            rul = float(prediction[0][0])

            st.success(f"Predicted RUL: {rul:.2f} cycles")

            if len(engine_data) < SEQUENCE_LENGTH:
                st.warning(
                    f"This engine only had {len(engine_data)} cycles of data "
                    f"(model expects {SEQUENCE_LENGTH}); the sequence was zero-padded, "
                    "so the prediction may be less reliable."
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a CSV with the engine's sensor readings to get a prediction.")
   
