#!/usr/bin/env python
# coding: utf-8

# In[31]:


import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import StringIO

# Streamlit App Title
st.title("Stock Price Prediction App 📈")

# Description
st.write("""
This app allows you to upload stock price data (CSV format) and predict future prices using Facebook's Prophet model.
You can also download the forecasted data for further analysis.  
""")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    # Display the first few rows of the uploaded file
    st.write("### Uploaded Data:")
    st.write(df.head())

    # Check if 'Date' and 'Close' columns are present
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("The file must contain 'Date' and 'Close' columns.")
    else:
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Prepare data for forecasting
        forecast_df = df[['Date', 'Close']].copy()
        forecast_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        
        # Get the number of days to predict from user
        prediction_days = st.slider("Select the number of days to predict", min_value=1, max_value=365, value=5)
        
        # Build and fit the Prophet model
        model = Prophet(daily_seasonality=True, seasonality_mode='multiplicative')
        model.fit(forecast_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)
        
        # Display predictions for the future days
        last_date = forecast_df['ds'].max()
        predictions = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        st.write(f"### Forecast for the next {prediction_days} days:")
        st.write(predictions)
        
        # Plot the forecast
        fig1 = model.plot(forecast)
        plt.title('Forecasted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        st.pyplot(fig1)
        
        # Option to download forecasted data
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name='forecasted_stock_prices.csv',
            mime='text/csv'
        )


# In[ ]:




