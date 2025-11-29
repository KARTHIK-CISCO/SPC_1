import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import joblib

st.set_page_config(page_title="Stock Forecast - Option A", layout="wide")

# --------------------------- Helpers ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("AAPL.csv"):
        df = pd.read_csv("AAPL.csv")
    else:
        st.error("No CSV uploaded and AAPL.csv not found in working dir.")
        return None

    # require Date and Close columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df = df[['Date','Close']].reset_index(drop=True)
    return df


def create_sequences(values, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i, 0])
        y.append(values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


@st.cache_resource(show_spinner=False)
def build_and_train_lstm(X_train, y_train, epochs=30, batch_size=32):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05, shuffle=False, verbose=0)
    return model


def iterative_forecast(model, last_seq_scaled, steps, scaler):
    # last_seq_scaled shape (seq_len, 1) or (1, seq_len, 1)
    seq = last_seq_scaled.copy().reshape(1, last_seq_scaled.shape[0], 1)
    preds = []
    for _ in range(steps):
        p = model.predict(seq, verbose=0)[0][0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    preds = np.array(preds).reshape(-1,1)
    return scaler.inverse_transform(preds)


# --------------------------- Sidebar ---------------------------
st.sidebar.header("Deployment Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Close)", type=['csv'])
seq_len = st.sidebar.number_input("Sequence length (LSTM lookback)", min_value=10, max_value=180, value=60, step=10)
max_horizon = st.sidebar.slider("Max future forecast days", min_value=1, max_value=90, value=30)
train_epochs = st.sidebar.number_input("Training epochs (if retrain)", min_value=1, max_value=200, value=30)
retrain = st.sidebar.checkbox("Retrain LSTM (if you want)", value=False)
load_saved_model = st.sidebar.checkbox("Load saved LSTM model if exists (lstm_model.h5)", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Write requirements.txt to disk"):
    reqs = [
        "streamlit\n",
        "pandas\n",
        "numpy\n",
        "scikit-learn\n",
        "tensorflow\n",
        "matplotlib\n",
        "joblib\n",
        "statsmodels\n"
    ]
    with open('requirements.txt','w') as f:
        f.writelines(reqs)
    st.sidebar.success("requirements.txt written to disk")


# --------------------------- Main ---------------------------
st.title("Option A — LSTM Deployment (Selectable Day + Up to 90-day Forecast)")

df = load_csv(uploaded_file)
if df is None:
    st.stop()

st.subheader("Data preview")
st.dataframe(df.tail(5))

last_date = df['Date'].max()
st.caption(f"Dataset ends on: {last_date.date()}")

# Prepare data
values = df[['Close']].values.astype('float32')
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Train/Test split (80/20)
train_size = int(len(values) * 0.8)

# Create sequences
X, y = create_sequences(scaled, seq_len=seq_len)
# Adjust train indices because sequences start at seq_len
adjusted_train_size = train_size - seq_len
if adjusted_train_size < 1:
    st.error("Not enough data for the chosen sequence length. Reduce sequence length or upload more data.")
    st.stop()

X_train, X_test = X[:adjusted_train_size], X[adjusted_train_size:]
y_train, y_test = y[:adjusted_train_size], y[adjusted_train_size:]

# Load or train model
model = None
if load_saved_model and os.path.exists('lstm_model.h5'):
    try:
        model = load_model('lstm_model.h5')
        st.success("Loaded saved LSTM model from lstm_model.h5")
    except Exception as e:
        st.warning(f"Failed to load saved model: {e}")
        model = None

if model is None:
    if retrain:
        with st.spinner("Training LSTM model — this may take a while depending on data/epochs..."):
            model = build_and_train_lstm(X_train, y_train, epochs=int(train_epochs))
            model.save('lstm_model.h5')
            st.success("Training complete — model saved as lstm_model.h5")
    else:
        # If not retrain and no saved model, train quickly with small epochs
        with st.spinner("Training LSTM model (default quick training, you can retrain with sidebar)..."):
            model = build_and_train_lstm(X_train, y_train, epochs=10)
            model.save('lstm_model.h5')
            st.info("Quick model trained and saved as lstm_model.h5 — consider retraining with more epochs for better accuracy")

# Evaluate on test set
with st.spinner("Evaluating model on test set..."):
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))

    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    try:
        test_r2 = r2_score(y_true, y_pred)
    except:
        test_r2 = float('nan')

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("LSTM RMSE", f"{test_rmse:.4f}")
col2.metric("LSTM MAPE (%)", f"{test_mape:.3f}")
col3.metric("LSTM R²", f"{test_r2:.4f}")

st.markdown("---")

# --------------------------- User Forecast Inputs ---------------------------
st.header("Make a prediction for a specific date")
selected_date = st.date_input("Select a date (past or future)", value=last_date.date())
selected_date = pd.to_datetime(selected_date)

horizon_days = st.slider("Forecast horizon (days ahead when selecting future dates)", min_value=1, max_value=90, value=30)

# If selected_date is after last_date, compute days ahead
days_ahead = (selected_date - last_date).days

# Predict for selected_date
if days_ahead <= 0:
    # historical or last_date: find index in df and predict next value using sequence that ends at that day
    idx = df.index[df['Date'] == selected_date]
    if len(idx) == 0:
        st.warning("Selected historical date not found in dataset. Pick a date that exists in the uploaded CSV or choose a future date.")
    else:
        i = idx[0]
        if i - seq_len < 0:
            st.warning("Not enough prior history to make a prediction for this date with current sequence length.")
        else:
            seq_values = scaled[i-seq_len:i]
            seq_values = seq_values.reshape(1, seq_len, 1)
            pred_scaled = model.predict(seq_values, verbose=0)[0][0]
            pred = scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]
            actual = df.loc[i,'Close']

            st.subheader(f"Prediction for {selected_date.date()}")
            col_a, col_b = st.columns([1,2])
            col_a.metric("Actual", f"{actual:.4f}")
            col_a.metric("Predicted", f"{pred:.4f}")

            # small plot: actual vs predicted (single point plotted with neighbors)
            neigh = 5
            start = max(0, i-neigh)
            end = min(len(df)-1, i+neigh)
            fig, ax = plt.subplots(figsize=(3.5,2))
            ax.plot(df['Date'].iloc[start:end+1], df['Close'].iloc[start:end+1], label='Actual')
            ax.scatter([selected_date], [actual], color='green', label='Actual (selected)')
            ax.scatter([selected_date], [pred], color='red', label='Predicted')
            ax.set_title('Actual vs Pred (small)')
            ax.tick_params(axis='x', labelrotation=45, labelsize=6)
            ax.legend(fontsize=6)
            st.pyplot(fig)

else:
    # Future date selected
    if days_ahead > 150:
        st.error("This app supports forecasting up to 90 days into the future. Choose a nearer date.")
    else:
        # iterative forecast up to days_ahead (or up to horizon_days whichever smaller)
        steps = min(days_ahead, horizon_days, 90)
        last_seq = scaled[-seq_len:]
        future_preds = iterative_forecast(model, last_seq, steps, scaler)
        pred_for_date = future_preds[days_ahead-1][0] if days_ahead-1 < len(future_preds) else None

        if pred_for_date is None:
            st.error("Could not forecast for selected date — please try a nearer date or increase horizon.")
        else:
            st.subheader(f"Prediction for {selected_date.date()} ( {days_ahead} days ahead )")
            col_a, col_b = st.columns([1,2])
            col_a.metric("Predicted", f"{pred_for_date:.4f}")

            # small selected-day plot: show last 10 days + 30 days forecast
            recent_days = 10
            recent_dates = df['Date'].iloc[-recent_days:]
            recent_actuals = df['Close'].iloc[-recent_days:]
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_preds))]

            fig, ax = plt.subplots(figsize=(3.5,2))
            ax.plot(recent_dates, recent_actuals, label='Recent Actuals')
            ax.plot(future_dates, future_preds.flatten(), label='Future Preds')
            ax.axvline(last_date, color='gray', linestyle='--')
            ax.set_title('Recent + Future (small)')
            ax.tick_params(axis='x', labelrotation=45, labelsize=6)
            ax.legend(fontsize=6)
            st.pyplot(fig)

# --------------------------- 30-day small forecast sparkline ---------------------------
st.markdown("---")
st.subheader("30-day small forecast")
with st.spinner("Computing 30-day forecast..."):
    last_seq = scaled[-seq_len:]
    spark_preds = iterative_forecast(model, last_seq, 30, scaler)
    spark_dates = [last_date + timedelta(days=i+1) for i in range(30)]

fig2, ax2 = plt.subplots(figsize=(8,2))
ax2.plot(spark_dates, spark_preds.flatten())
ax2.set_title('30-day Forecast (small)')
ax2.tick_params(axis='x', labelrotation=45, labelsize=8)
st.pyplot(fig2)

# --------------------------- Full forecast up to user-selected days ---------------------------
st.markdown("---")
st.header("Full forecast (select up to 90 days)")
full_days = st.slider("Select days to forecast (full view)", min_value=1, max_value=90, value=30)

with st.spinner("Computing full forecast..."):
    full_preds = iterative_forecast(model, scaled[-seq_len:], full_days, scaler)
    full_dates = [last_date + timedelta(days=i+1) for i in range(full_days)]

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df['Date'], df['Close'], label='Historical')
ax3.plot(full_dates, full_preds.flatten(), label='Forecast')
ax3.set_title(f'Historical + {full_days}-day Forecast')
ax3.legend()
st.pyplot(fig3)

st.markdown("---")
st.caption("Notes: The app trains a LSTM model on the provided CSV. For production use, train offline and provide a saved model (lstm_model.h5) to the working directory, or enable retrain and increase epochs in the sidebar.")

# Provide download links for forecasts
if st.button("Download 30-day forecast CSV"):
    out_df = pd.DataFrame({'Date': spark_dates, 'Predicted_Close': spark_preds.flatten()})
    out_csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("Click to download", data=out_csv, file_name='30_day_forecast.csv', mime='text/csv')

# End of app
