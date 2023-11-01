

import datetime
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from app import init_app
from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.instrument.model import Instrument

from statsmodels.tsa.vector_ar.vecm import coint_johansen

__SUPPORTED_INSTRUMENTS = [
    Instrument(symbol='IYE'),
    Instrument(symbol='VDE'),
    Instrument(symbol='XLE'),
    Instrument(symbol='FENY'),
    Instrument(symbol='XES'),
    Instrument(symbol='EWC'),
    Instrument(symbol='EWA')
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
input_completed = len(chosen_instruments) >= 2
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
if not input_completed:
    st.stop()
backtest_container = st.container()

backtest_container.header("Backtest")
# TODO: Setup flow for date range backtest
# date_range = st.date_input("Date range")
historical_data = app.get_historical_data(
    instruments=chosen_instruments, start_date=datetime.datetime(2015, 1, 1))
start_date, end_date = pd.Timestamp(
    historical_data.index.min()), pd.Timestamp(historical_data.index.max())

in_sample_dataset, out_sample_dataset = train_test_split(
    historical_data, test_size=0.3, shuffle=False)

backtest_container.metric(
    "Date range", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}", f"{end_date - start_date}")
backtest_container.line_chart(historical_data)

insample_section, outsample_section = backtest_container.tabs(
    ['IS Summary', 'OS Summary'])
# In sample section
insample_section.subheader("Input")
num_folds = insample_section.slider("Number of folds", 3, 5)

insample_section.subheader("Cross validation")
folds = app.split_historical_data(
    historical_data=historical_data, num_folds=num_folds)
for i, tab in enumerate(insample_section.tabs([f"Fold {i+1}" for i in range(num_folds)])):
    train_data, valid_data = folds[i]
    fold_data = [train_data, valid_data]
    labels = ['Train', 'Valid']

    for j in range(len(labels)):
        data = fold_data[j]
        start_date, end_date = pd.Timestamp(
            data.index.min()), pd.Timestamp(data.index.max())
        tab.metric(f"{labels[j]} Date Range", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}",
                   f"{end_date - start_date}")

    for j, subtab in enumerate(tab.tabs(labels)):
        data = fold_data[j]
        # Cointegration statistic
        subtab.text("Cointegration statistic")
        jres = coint_johansen(data, det_order=0, k_ar_diff=1)
        coint_result = pd.DataFrame(np.concatenate([jres.max_eig_stat.reshape(-1, 1), jres.max_eig_stat_crit_vals], axis=1),
                                    columns=['Statistic', 'Crit 90%', 'Crit 95%', 'Crit 99%'])
        subtab.table(coint_result)
        hedge_ratio, spread, spread_std = app.mean_reversion_strategy.calc_trading_inputs(
            price=data)
        # Result
        subtab.text("Result")
        try:
            ret, sharp_ratio, max_drawdown_deep, max_drawdown_duration = app.mean_reversion_strategy.trade(
                data, hedge_ratio, spread, spread_std)
            subtab.metric("Sharp ratio", round(sharp_ratio, 2))
            col_max_drawdown_deep, col_max_drawdown_duration = subtab.columns(
                2)
            col_max_drawdown_deep.metric("Max drawdown",
                                         f"-{round(max_drawdown_deep['drawdown_deep'], 2)}%")
            col_max_drawdown_duration.metric("Max drawdown duration",
                                             f"{max_drawdown_duration['drawdown_duration']}")
            metric_detail = subtab.expander("Detail")
            with metric_detail:
                col1, col2 = metric_detail.columns(2)
            with col1:
                col1.text("Max drawdown")
                col1.write(max_drawdown_deep)
            with col2:
                col2.text("Max drawdown duration")
                col2.write(max_drawdown_duration)
            subtab.line_chart(ret.dropna().cumsum())
        except ValueError as err:
            subtab.warning(err)

        # Spread & Std
        subtab.text("Spread & Std")

        spread_summary = subtab.expander("Summary")
        with spread_summary:
            col1, col2 = spread_summary.columns(2)
            with col1:
                col1.text("Spread")
                col1.text(spread.describe())
            with col2:
                col2.text("Spread Std")
                col2.text(spread_std.describe())

        subtab.line_chart(pd.DataFrame({
            'spread': spread,
            'std': spread_std,
            '-std': -spread_std
        }))


outsample_section.subheader("Out sample test")
