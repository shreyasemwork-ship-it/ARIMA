import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIMA Stock Forecasting", layout="wide")

st.title("📈 ARIMA Stock Forecasting App")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    try:
        st.info("Downloading 5 years of data from Yahoo Finance...")

        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"]

        st.subheader("Historical Data")
        st.dataframe(data.tail())

        st.info("Training ARIMA Model...")

        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()

        forecast_end = pd.Timestamp("2027-06-30")

        days_to_forecast = (
            forecast_end - close_prices.index[-1]
        ).days

        forecast = model_fit.forecast(
            steps=days_to_forecast
        )

        future_dates = pd.date_range(
            start=close_prices.index[-1] + pd.Timedelta(days=1),
            periods=days_to_forecast,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted_Close": forecast.values
        })

        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df.tail(20)
        )

        fig, ax = plt.subplots(figsize=(12,6))

        ax.plot(
            close_prices.index,
            close_prices,
            label="Historical"
        )

        ax.plot(
            forecast_df["Date"],
            forecast_df["Forecasted_Close"],
            label="Forecast"
        )

        ax.set_title(
            f"{ticker} Forecast till June 2027"
        )

        ax.legend()

        st.pyplot(fig)

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "Download Forecast CSV",
            csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(str(e))
