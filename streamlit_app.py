import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import streamlit as st

# Title and description
st.title("Samsung Stock Price Prediction")
st.write("""
### Predict stock prices for the next 2 years using XGBoost with a 5-year lookback period.
The dataset is preloaded from the same directory as this script.
""")

# File path for the dataset
file_path = "samsung_stock_data.xlsx"

try:
    # Load the dataset
    stock_data = pd.read_excel(file_path)

    # Convert 'Date' column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    stock_data = stock_data.dropna(subset=['Date'])  # Drop invalid dates
    stock_data_filtered = stock_data[(stock_data['Date'] >= '2000-01-01') & (stock_data['Date'] <= '2023-12-31')]

    # Select and scale the 'Close' prices
    close_prices = stock_data_filtered['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare data with lookback period
    lookback = 1825  # 5 years (approx.)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape for XGBoost
    X_train_flat = X.reshape(X.shape[0], -1)

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01)
    xgb_model.fit(X_train_flat, y)

    # Predict the next 2 years
    last_known_data = scaled_data[-lookback:]
    predicted_prices = []

    for _ in range(730):  # Predict for 2 years
        input_data = last_known_data.reshape(1, lookback, 1)
        prediction = xgb_model.predict(input_data.flatten().reshape(1, -1))
        predicted_prices.append(prediction[0])
        last_known_data = np.append(last_known_data[1:], prediction)

    # Rescale predictions back to original scale
    predicted_prices_rescaled = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Generate future dates
    last_date = stock_data_filtered['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=730)

    # Plot results
    st.write("### Prediction Results")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stock_data_filtered['Date'], stock_data_filtered['Close'], label='Actual Prices', color='blue')
    ax.plot(future_dates, predicted_prices_rescaled, label='Predicted Prices (XGBoost)', color='orange', linestyle='--')
    ax.set_title("Samsung Stock Price Prediction for the Next 2 Years (XGBoost)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (KRW)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
