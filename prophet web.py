import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Stock Price Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your stock price CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file, encoding='UTF-8-SIG')

    # Remove any leading or trailing spaces from column names
    df.columns = [col.strip() for col in df.columns]

    # Display the first few rows of the dataset
    st.write("Uploaded Data:")
    st.write(df.head())
    
    # Check if 'Date' and 'Close' columns are present
    if 'Date' not in df.columns:
        st.error("The CSV file must contain a 'Date' column.")
    else:
        # Convert Date column to datetime
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        except:
            st.error("Date column format should be 'dd-mm-YYYY'.")
        
        # Find the 'Close' column
        close_col = None
        for col in df.columns:
            if col.lower().strip() == 'close':
                close_col = col
                break
        
        if close_col is None:
            st.error("The CSV file must contain a 'Close' column.")
        else:
            # Prepare data for Prophet
            forecast_df = df[['Date', close_col]].copy()
            forecast_df.rename(columns={'Date': 'ds', close_col: 'y'}, inplace=True)

            # Ensure the dataframe is sorted by date
            forecast_df = forecast_df.sort_values(by='ds')

            # Display last date in the dataset
            last_date = forecast_df['ds'].max()
            st.write(f"Last date in the dataset: {last_date.date()}")

            # Initialize and fit the Prophet model
            model = Prophet(daily_seasonality=True, seasonality_mode='multiplicative')
            model.fit(forecast_df)

            # Create dataframe for future predictions
            future = model.make_future_dataframe(periods=5)

            # Make predictions
            forecast = model.predict(future)

            # Filter predictions for only future dates
            predictions = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            st.write("Forecast for the next 5 days:")
            st.write(predictions)

            # Plotting the forecast
            fig1 = model.plot(forecast)
            plt.title('Forecast for the Next 5 Days')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            st.pyplot(fig1)
            
            st.success("Forecasting complete.")
