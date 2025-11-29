import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Stock Forecast", layout="wide")

# --------------------------- Helper Functions ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Please upload a CSV file containing Date & Close columns.")
        return None

    if "Date" not in df.columns or "Close" not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def create_sequences(values, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i, 0])
        y.append(values[i, 0])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)

def build_and_train_lstm(X_train, y_train, epochs=25, batch_size=32):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.05, shuffle=False, verbose=0)
    return model

def iterative_forecast(model, last_seq_scaled, steps, scaler):
    seq = last_seq_scaled.reshape(1, -1, 1)
    preds_scaled = []
    for _ in range(steps):
        pred = model.predict(seq, verbose=0)[0][0]
        preds_scaled.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled)

# --------------------------- Main App ---------------------------
st.title("📈 LSTM Stock Forecast with Prediction Graph")

uploaded_file = st.file_uploader("Upload CSV (must include Date & Close)", type=["csv"])
df = load_csv(uploaded_file)
if df is None:
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.tail())

last_date = df["Date"].max()
st.write(f"**Dataset ends on:** {last_date.date()}")

# Prepare data
values = df[["Close"]].values.astype("float32")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

seq_len = 60
X, y = create_sequences(scaled, seq_len)

train_size = int(len(values) * 0.8)
adj_train_size = train_size - seq_len

X_train, X_test = X[:adj_train_size], X[adj_train_size:]
y_train, y_test = y[:adj_train_size], y[adj_train_size:]

# --------------------------- Model Training Logic ---------------------------
st.header("Model Training")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None

if st.button("🔄 Retrain LSTM Model"):
    with st.spinner("Training LSTM model..."):
        st.session_state.model = build_and_train_lstm(X_train, y_train, epochs=25)
    st.session_state.model_trained = True
    st.success("✔ Model retrained successfully!")

if not st.session_state.model_trained:
    st.warning("⚠ Model is not trained yet. Click 'Retrain LSTM Model' to train.")
    st.stop()

model = st.session_state.model

# --------------------------- Evaluation ---------------------------
st.header("Model Performance")

y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
try:
    r2 = r2_score(y_true, y_pred)
except:
    r2 = float("nan")

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.4f}")
col2.metric("MAPE (%)", f"{mape:.3f}")
col3.metric("R² Score", f"{r2:.4f}")

# --------------------------- Predict Any Date ---------------------------
st.header("Predict Any Date")

selected_date = st.date_input("Select a date (past or future)", value=last_date.date())
selected_date = pd.to_datetime(selected_date)
days_ahead = (selected_date - last_date).days

if days_ahead <= 0:
    idx = df.index[df["Date"] == selected_date]
    if len(idx) == 0:
        st.error("Historical date not found in dataset.")
    else:
        i = idx[0]
        if i - seq_len < 0:
            st.error("Not enough data before this date.")
        else:
            seq_vals = scaled[i - seq_len:i].reshape(1, seq_len, 1)
            pred_scaled = model.predict(seq_vals, verbose=0)[0][0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            actual = df.loc[i, "Close"]

            st.subheader(f"Prediction for {selected_date.date()}")
            st.write(f"**Actual:** {actual:.4f}")
            st.write(f"**Predicted:** {pred:.4f}")

elif days_ahead > 0:
    if days_ahead > 90:
        st.error("Forecast beyond 90 days is not supported.")
    else:
        future_preds = iterative_forecast(model, scaled[-seq_len:], days_ahead, scaler)
        pred_val = future_preds[days_ahead - 1][0]
        st.subheader(f"Prediction for {selected_date.date()} ({days_ahead} days ahead)")
        st.write(f"**Predicted:** {pred_val:.4f}")

# --------------------------- Prediction Graph ---------------------------
st.header("Prediction Graph")

if days_ahead <= 0 and len(idx) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[selected_date],
        y=[actual],
        mode='markers+lines',
        name='Actual',
        marker=dict(color='red', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[selected_date],
        y=[pred],
        mode='markers+lines',
        name='Predicted',
        marker=dict(color='red', size=10)
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Close Price on {selected_date.date()}",
        xaxis_title="Date",
        yaxis_title="Close Price",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

elif days_ahead > 0 and days_ahead <= 90:
    fig = go.Figure()
    past_dates = df["Date"].iloc[-30:]
    past_values = df["Close"].iloc[-30:]
    fig.add_trace(go.Scatter(x=past_dates, y=past_values, mode='lines', name='Historical Close'))

    future_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds[:days_ahead].flatten(),
                             mode='lines+markers', name='Predicted'))

    fig.update_layout(
        title=f"Forecast up to {selected_date.date()}",
        xaxis_title="Date",
        yaxis_title="Close Price",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------- 30-Day Forecast Download ---------------------------
st.header("Download 30-Day Forecast")

future30 = iterative_forecast(model, scaled[-seq_len:], 30, scaler)
dates30 = [last_date + timedelta(days=i + 1) for i in range(30)]
df30 = pd.DataFrame({"Date": dates30, "Predicted_Close": future30.flatten()})
csv30 = df30.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download 30-Day Forecast CSV",
    data=csv30,
    file_name="30_day_forecast.csv",
    mime="text/csv"
)

st.success("App ready. Forecasting complete.")
