import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction using Tuned LSTM Model")


# ======================================================================
# Function to create sequences
# ======================================================================
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ======================================================================
# File Upload
# ======================================================================
uploaded_file = st.file_uploader("Upload stock dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Date" not in df.columns or "Close" not in df.columns:
        st.error("Dataset must contain 'Date' and 'Close' columns!")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    data = df[["Close"]].values

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)

    # Train-test split
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
    X_test = X_test.reshape((X_test.shape[0], seq_len, 1))

    st.subheader("🔧 Training Tuned LSTM Model...")

    # ======================================================================
    # TUNED LSTM MODEL
    # ======================================================================
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),

        LSTM(64),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    st.success("Tuned LSTM Model Trained Successfully! 🎉")

    # Predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    st.subheader("📉 Model Performance")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.4f}")

    # ======================================================================
    # Plot Actual vs Predicted
    # ======================================================================
    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Date"].iloc[-len(y_true):], y_true, label="Actual", linewidth=2)
    ax.plot(df["Date"].iloc[-len(y_pred):], y_pred, label="Predicted", linewidth=2)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)


    # ======================================================================
    # 30-Day Forecast
    # ======================================================================
    st.subheader("🔮 30-Day Future Forecast")

    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    lstm_forecast = []

    for _ in range(30):
        pred = model.predict(last_60)[0][0]
        lstm_forecast.append(pred)
        last_60 = np.append(last_60[:, 1:, :], [[[pred]]], axis=1)

    forecast_30 = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "Day": np.arange(1, 31),
        "Forecasted Price": forecast_30.flatten()
    })

    st.dataframe(forecast_df)

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(forecast_df["Day"], forecast_df["Forecasted Price"], marker="o")
    ax2.set_title("30-Day Future Forecast")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Predicted Price")
    st.pyplot(fig2)

    # ======================================================================
    # Download Forecast CSV
    # ======================================================================
    st.download_button(
        label="📥 Download 30-Day Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name="30_day_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a dataset to start prediction.")
