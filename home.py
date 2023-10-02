import streamlit as st
import requests
import io
import os
import pandas as pd
import datetime
import numpy as np

from statsmodels.tsa.vector_ar.vecm import coint_johansen

START_HISTORICAL_DATE = datetime.datetime(2015, 1, 1)
LOOK_BACK = 46

st.title("Next's positions")


@st.cache_data
def load_historical_data():
    """
        Load data
    """
    symbols = ['IYE', 'VDE']
    now = datetime.datetime.now()
    data_dir = f"data/{now.strftime('%Y%m%d')}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    raw_data = {}
    for symbol in symbols:
        file_name = f"{data_dir}/{symbol}.csv"
        try:
            data = pd.read_csv(file_name, delimiter=",")
        except IOError as err:
            print(err)
            url = "https://api.twelvedata.com/time_series" \
                "?apikey=70921edd203f44aab576fefe41befc92" \
                f"&symbol={symbol}" \
                "&interval=1day&type=etf&format=CSV" \
                f"&start_date={START_HISTORICAL_DATE.strftime('%Y-%m-%d')}"
            response = requests.get(url, timeout=60)
            data = pd.read_csv(io.StringIO(
                response.content.decode('utf-8')), delimiter=";")
            data.to_csv(file_name, mode='w', index=False)
        data = data.set_index("datetime")
        raw_data[symbol] = data
    return raw_data


@st.cache_data
def get_hedge_ratio(close_price):
    """
        Calculate hedge ratio
    """
    hedge_ratio = pd.DataFrame(
        np.NaN, columns=close_price.columns, index=close_price.index)
    for i in range(LOOK_BACK-1, len(close_price)):
        price_period = close_price[i-LOOK_BACK+1:i+1]
        jres = coint_johansen(price_period, det_order=0, k_ar_diff=1)
        hedge_ratio.iloc[i] = jres.evec.T[0]
    return hedge_ratio


historical_data_map = load_historical_data()
# Extract data
close_price = {}
for symbol, symbol_data in historical_data_map.items():
    close_price[symbol] = symbol_data["close"]
close_price = pd.DataFrame(close_price).sort_index()
hedge_ratio = get_hedge_ratio(close_price=close_price)
mkt_value = hedge_ratio * close_price
spread = mkt_value.sum(axis=1)
zscore_spread = (spread - spread.rolling(LOOK_BACK).mean()) / \
    spread.rolling(LOOK_BACK).std()
st.line_chart(zscore_spread.dropna().tail(20))
