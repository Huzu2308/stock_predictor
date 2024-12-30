import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Streamlit app title
st.title("Samsung Stock Price Prediction")

# Step 1: Load the data from the same directory
file_name = "samsung_stock_data.xlsx"  # The Excel file is in the same directory as the script
stock_data = pd.read_excel(file_name)

# Step 2: Convert the 'Date' column to datetime and filter the required date range
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')  # Handles invalid dates
stock_data = stock_data.dropna(subset=['Date'])  # Drop rows with invalid dates
stock_data_filtered = stock_data[(stock_data['Date'] >= '2000-01-01') & (stock_data['Date'] <= '2023-12-31')]

# Step 3: Select and scale the 'Close' prices
close_prices = stock_data_filtered['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Step 4: Create the training dataset with a lookback of 365 days (1 year)
lookback = 365  # 1 year of past data
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])  # Use the past 'lookback' days
    y.append(scaled_data[i, 0])  # Predict the next day
X, y = np.array(X), np.array(y)

# Step 5: Flatten X for XGBoost
X_train_flat = X.reshape(X.shape[0], -1)

# Step 6: Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_train_flat, y, test_size=0.2, shuffle=False)

# Step 7: Train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train, y_train)

# Step 8: Predict the next day's price for each of the next 730 days (2 years)
last_known_data = scaled_data[-lookback:]  # Use the last 'lookback' days from the actual data
predicted_prices = []

for _ in range(730):  # Predict the next 730 days (2 years)
    input_data = last_known_data.reshape(1, lookback, 1)  # Reshape for XGBoost input
    prediction = xgb_model.predict(input_data.flatten().reshape(1, -1))  # Make prediction
    predicted_prices.append(prediction[0])  # Add predicted price to the list
    last_known_data = np.append(last_known_data[1:], prediction)  # Slide the window

# Step 9: Convert predictions back to the original scale
predicted_prices_rescaled = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Step 10: Generate future dates for the next 2 years (730 days)
last_date = stock_data_filtered['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=730)

# Step 11: Plotting the actual and predicted prices
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the actual prices (only until the last historical date)
ax.plot(stock_data_filtered['Date'], stock_data_filtered['Close'], label='Actual Prices', color='blue')

# Plot the predicted prices (only after the last historical date)
ax.plot(future_dates, predicted_prices_rescaled, label='Predicted Prices', color='red', linestyle='--')

ax.set_title("Samsung Stock Price Prediction for the Next 2 Years (XGBoost) with 1-Year Lookback")
ax.set_xlabel("Date")
ax.set_ylabel("Price (KRW)")
ax.legend()
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)
