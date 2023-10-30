

import streamlit as st
from app import init_app
from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.instrument.model import Instrument

__SUPPORTED_INSTRUMENTS = [
    Instrument(symbol='IYE'),
    Instrument(symbol='VDE'),
    Instrument(symbol='XLE'),
    Instrument(symbol='FENY'),
    Instrument(symbol='XES')
]


app = init_app()
# app.fetch_session()

st.title("Mean Reversion Strategy")
st.info("""
        Mean reversion is a theory used in finance that suggests that 
        asset prices and historical returns tend to move towards their long-term average levels.
        """)

# Input parameter section
st.header("Input parameters")
chosen_instruments = st.multiselect("Instruments", __SUPPORTED_INSTRUMENTS)
st.divider()
# Live trading section
live_trading_container = st.container()
live_trading_container.header("Live trading")
auth_err = app.check_auth()
if auth_err:
    live_trading_container.write(f"Got auth error: {auth_err}")
    live_trading_container.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
else:
    # Portfolio section
    # Order section
    live_trading_container.subheader("Order")
    with live_trading_container.expander("Order section"):
        st.write("Yeah")

st.divider()
# Backtest section
backtest_container = st.container()

backtest_container.header("Backtest")
# TODO: Setup flow for date range backtest
# date_range = st.date_input("Date range")
historical_data = app.get_historical_data(instruments=chosen_instruments)
backtest_container.text(
    f"Time range: {historical_data.index.min()} - {historical_data.index.max()}")
backtest_container.line_chart(historical_data)

backtest_container.subheader("In sample dataset")
num_folds = backtest_container.slider("Number of folds", 3, 5)

for tab in backtest_container.tabs([f"Fold {i+1}" for i in range(num_folds)]):
    tab.write("Hello")

backtest_container.subheader("Out sample test")
