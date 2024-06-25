import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import load_model  # Correct import for TensorFlow Keras
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("Stock Price Predictor App")

# User Input for Stock ID
stock = st.text_input("Enter the stock id", "GOOG")

# Date Range for Data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download Stock Data
google_data = yf.download(stock, start=start, end=end)

# Load the Pre-trained Model
model_path = "Latest_stock_price_model.keras"  # Ensure this path is correct
try:
    model = load_model(model_path)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"File not found: {model_path}")
except ValueError as e:
    st.error(f"Error loading model: {e}")

# Display Stock Data
st.subheader("Stock Data")
st.write(google_data)

# Splitting Data for Training and Testing
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Close'][splitting_len:])

# Function to Plot Graph
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plot Moving Averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare Data for Prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make Predictions
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Plot Predictions
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len+100:])

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader("Original Close price vs Predicted Close price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data['Close'][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data not used", "Original test data", "Predicted test data"])
st.pyplot(fig)
