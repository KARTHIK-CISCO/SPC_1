import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Forecast App (LSTM)", initial_sidebar_state="expanded")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        # try local file name fallback
        try:
            df = pd.read_csv("AAPL.csv")
        except Exception:
            st.error("No dataset uploaded and AAPL.csv not found. Please upload a CSV with Date and Close columns.")
            return None
    else:
        df = pd.read_csv(uploaded_file)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns or 'Close' not in df.columns:
        # try common variations
        possible_date = [c for c in df.columns if 'date' in c.lower()]
        possible_close = [c for c in df.columns if 'close' in c.lower() or 'adj close' in c.lower() or 'adj_close' in c.lower()]
        if possible_date and possible_close:
            df = df[[possible_date[0], possible_close[0]]].rename(columns={possible_date[0]:'Date', possible_close[0]:'Close'})
        else:
            st.error("CSV must contain 'Date' and 'Close' (or similar).")
            return None
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def train_sarima(train_values, order=(2,1,2), seasonal_order=(1,1,1,12)):
    model = SARIMAX(train_values, order=order, seasonal_order=seasonal_order)
    fit = model.fit(disp=False)
    return fit

@st.cache_resource
def train_lstm_model(values, seq_len=60, epochs=30, verbose=0):
    """
    Train LSTM and return (model, scaler, X_train, X_test, y_train, y_test)
    Caching avoids re-train when data unchanged.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1,1))

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Fit
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, shuffle=False, verbose=verbose)
    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "seq_len": seq_len
    }

def inverse_transform_array(scaler, arr):
    return scaler.inverse_transform(np.array(arr).reshape(-1,1)).flatten()

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2

def forecast_lstm_sequence(model, last_seq_scaled, steps, scaler):
    # last_seq_scaled shape: (1, seq_len, 1)
    out = []
    seq = last_seq_scaled.copy()
    for _ in range(steps):
        pred_scaled = model.predict(seq, verbose=0)[0][0]
        out.append(pred_scaled)
        # append and shift
        seq = np.append(seq[:,1:,:], [[[pred_scaled]]], axis=1)
    return inverse_transform_array(scaler, out)

# ---------------------------
# Sidebar: Inputs
# ---------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Close)", type=['csv'])
df = load_data(uploaded_file)
if df is None:
    st.stop()

model_choice = st.sidebar.selectbox("Model choice", ["LSTM only", "SARIMA + LSTM"])
seq_len = st.sidebar.number_input("Sequence length for LSTM (days)", min_value=10, max_value=120, value=60, step=5)
epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=200, value=30, step=1)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 90, 30)
specific_date = st.sidebar.date_input("Select a specific date to inspect (existing or up to future forecast)", value=df['Date'].iloc[-1].date())
run_button = st.sidebar.button("Run / Update")

# Always show some basic dataset info
st.sidebar.markdown(f"**Dataset Range:** {df['Date'].min().date()}  →  {df['Date'].max().date()}")
st.sidebar.markdown(f"**Total rows:** {len(df)}")

# ---------------------------
# Main
# ---------------------------
st.title("Time Series Forecasting (LSTM ± SARIMA)")
st.write("Upload your CSV or use the default `AAPL.csv`. Choose model, seq length, epochs, and forecast horizon (max 90).")

if run_button:
    with st.spinner("Training models and producing forecasts... this may take a little while"):
        values = df['Close'].values.astype(float)

        # Train SARIMA on train portion if chosen
        train_size = int(len(values) * 0.8)
        train_vals = values[:train_size]
        test_vals = values[train_size:]

        sarima_fit = None
        if model_choice != "LSTM only":
            try:
                sarima_fit = train_sarima(train_vals.reshape(-1))
            except Exception as e:
                st.warning(f"SARIMA failed: {e}")
                sarima_fit = None

        # Train LSTM (cached)
        lstm_res = train_lstm_model(values, seq_len=seq_len, epochs=epochs, verbose=0)
        model = lstm_res["model"]
        scaler = lstm_res["scaler"]
        X_test = lstm_res["X_test"]
        y_test = lstm_res["y_test"]

        # LSTM predictions on test set
        lstm_pred_scaled = model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

        lstm_rmse, lstm_mape, lstm_r2 = compute_metrics(y_true, lstm_pred)

        # SARIMA predict over test period
        sarima_pred = None
        if sarima_fit is not None:
            sarima_pred_vals = sarima_fit.predict(start=len(train_vals), end=len(values)-1)
            sarima_pred = np.array(sarima_pred_vals).reshape(-1)

            sarima_rmse, sarima_mape, sarima_r2 = compute_metrics(test_vals, sarima_pred)
        else:
            sarima_rmse = sarima_mape = sarima_r2 = None

        # TUNED results summary
        st.subheader("Model performance on test set")
        perf_cols = st.columns(3)
        perf_cols[0].metric("LSTM RMSE", f"{lstm_rmse:.5f}")
        perf_cols[1].metric("LSTM MAPE (%)", f"{lstm_mape:.3f}")
        perf_cols[2].metric("LSTM R²", f"{lstm_r2:.4f}")

        if sarima_fit is not None:
            st.text("SARIMA (if enabled):")
            sar_cols = st.columns(3)
            sar_cols[0].metric("SARIMA RMSE", f"{sarima_rmse:.5f}")
            sar_cols[1].metric("SARIMA MAPE (%)", f"{sarima_mape:.3f}")
            sar_cols[2].metric("SARIMA R²", f"{sarima_r2:.4f}")

        # Build aligned date index for test predictions
        # For LSTM we used seq_len so the test predictions correspond to dates starting from index (seq_len + split)
        total_len = len(values)
        X_full_len = len(values) - seq_len
        split_idx = int(X_full_len * 0.8)
        test_start_idx = seq_len + split_idx  # index in original values where test set starts
        test_dates = df['Date'].iloc[test_start_idx: test_start_idx + len(y_test)].reset_index(drop=True)

        # Create DataFrame showing actual vs predictions (test)
        test_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_true,
            'LSTM_Pred': lstm_pred
        })

        if sarima_pred is not None and len(sarima_pred) == len(test_df):
            test_df['SARIMA_Pred'] = sarima_pred

        # Forecast future days
        # LSTM multi-step forecast using last seq_len values
        last_seq = scaler.transform(values.reshape(-1,1))[-seq_len:].reshape(1, seq_len, 1)
        lstm_future = forecast_lstm_sequence(model, last_seq, forecast_days, scaler)

        # SARIMA future forecast
        sarima_future = None
        if sarima_fit is not None:
            try:
                sarima_future_vals = sarima_fit.forecast(forecast_days)
                sarima_future = np.array(sarima_future_vals).reshape(-1)
            except Exception:
                sarima_future = None

        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
        future_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM_Forecast': lstm_future
        })
        if sarima_future is not None:
            future_df['SARIMA_Forecast'] = sarima_future

        # ---------------------------
        # Specific date handling
        # ---------------------------
        selected_dt = pd.to_datetime(specific_date)
        st.header(f"Selected date: {selected_dt.date()}")

        # If selected date exists in historical df -> show actual and predicted (if in test)
        if selected_dt in df['Date'].values:
            idx = df.index[df['Date']==selected_dt][0]
            actual_val = df.loc[idx,'Close']
            in_test_range = (idx >= test_start_idx) and (idx < test_start_idx + len(y_test))
            if in_test_range:
                # map to test_df row
                row_idx = idx - test_start_idx
                pred_lstm_val = test_df.loc[row_idx,'LSTM_Pred']
                st.markdown(f"**Actual price:** {actual_val:.4f}")
                st.markdown(f"**LSTM Predicted (test):** {pred_lstm_val:.4f}")
                if 'SARIMA_Pred' in test_df.columns:
                    st.markdown(f"**SARIMA Predicted (test):** {test_df.loc[row_idx,'SARIMA_Pred']:.4f}")
            else:
                # If it's historical but not in test (train portion)
                st.markdown(f"**Actual price (train):** {actual_val:.4f}")
                st.markdown("Prediction not available (selected date is in training portion).")
        else:
            # If selected date is in future within forecast horizon, give predicted forecast value
            days_ahead = (selected_dt - df['Date'].iloc[-1]).days
            if days_ahead <= 0:
                st.markdown("Selected date is before the dataset end but not exactly present (maybe weekend). Closest market day predictions are shown below.")
            if 1 <= days_ahead <= forecast_days:
                pred_val = future_df.loc[days_ahead-1, 'LSTM_Forecast']
                st.markdown(f"**Predicted (LSTM) on {selected_dt.date()}:** {pred_val:.4f}")
                if sarima_future is not None:
                    st.markdown(f"**Predicted (SARIMA) on {selected_dt.date()}:** {future_df.loc[days_ahead-1, 'SARIMA_Forecast']:.4f}")
            else:
                st.markdown("Selected date is outside the future forecast horizon. Please choose a date within the next forecast horizon or a date present in dataset.")

        # ---------------------------
        # Small actual vs predicted plot centered on selected date
        # ---------------------------
        st.subheader("Compact: Actual vs Predicted (around selected date)")
        # pick a small window of +-7 days around selected date (if available in test_df), otherwise show last 30 days of test
        window = 7
        # Merge actual series and predictions across entire test range for indexing ease
        combined = test_df.copy()
        # If selected date in combined, center there else show last 30 rows
        if selected_dt in combined['Date'].values:
            mid_idx = combined.index[combined['Date']==selected_dt][0]
            start = max(mid_idx - window, 0)
            end = min(mid_idx + window + 1, len(combined))
            plot_df = combined.iloc[start:end]
        else:
            plot_df = combined.tail(30)

        fig1, ax1 = plt.subplots(figsize=(4,2.4))
        ax1.plot(plot_df['Date'], plot_df['Actual'], label='Actual', linewidth=1)
        ax1.plot(plot_df['Date'], plot_df['LSTM_Pred'], label='LSTM Pred', linewidth=1)
        if 'SARIMA_Pred' in plot_df.columns:
            ax1.plot(plot_df['Date'], plot_df['SARIMA_Pred'], label='SARIMA Pred', linewidth=1)
        ax1.set_xticks(plot_df['Date'][::max(1, len(plot_df)//4)])
        ax1.tick_params(axis='x', rotation=25, labelsize=7)
        ax1.set_ylabel("Price", fontsize=8)
        ax1.legend(fontsize=7, loc='best')
        plt.tight_layout()
        st.pyplot(fig1)

        # ---------------------------
        # Small future forecast (first 30 days or user chosen)
        # ---------------------------
        st.subheader("Compact: Future forecast (first 30 days or chosen horizon)")
        small_future_display = min(30, forecast_days)
        fig2, ax2 = plt.subplots(figsize=(4,2.2))
        ax2.plot(future_df['Date'].iloc[:small_future_display], future_df['LSTM_Forecast'].iloc[:small_future_display], label='LSTM Forecast', linewidth=1)
        if 'SARIMA_Forecast' in future_df.columns:
            ax2.plot(future_df['Date'].iloc[:small_future_display], future_df['SARIMA_Forecast'].iloc[:small_future_display], label='SARIMA Forecast', linewidth=1)
        ax2.set_xticks(future_df['Date'].iloc[:small_future_display][::max(1, small_future_display//4)])
        ax2.tick_params(axis='x', rotation=25, labelsize=7)
        ax2.set_ylabel("Price", fontsize=8)
        ax2.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)

        # ---------------------------
        # Full forecast and historical overlay (bigger)
        # ---------------------------
        st.subheader("Historical (last 120 days) + Forecast")
        hist_display = df.tail(120).copy()
        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(hist_display['Date'], hist_display['Close'], label='Historical', linewidth=1.5)
        ax3.plot(future_df['Date'], future_df['LSTM_Forecast'], label='LSTM Forecast', linestyle='--')
        if 'SARIMA_Forecast' in future_df.columns:
            ax3.plot(future_df['Date'], future_df['SARIMA_Forecast'], label='SARIMA Forecast', linestyle='--')
        ax3.legend()
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Price")
        plt.tight_layout()
        st.pyplot(fig3)

        # ---------------------------
        # Table outputs
        # ---------------------------
        st.subheader("Forecast table (first 30 rows shown)")
        st.dataframe(future_df.head(30))

        st.success("Done — interact with the controls in the sidebar to re-run with different settings.")
