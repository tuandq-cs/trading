

import datetime
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from app import init_app
from constants.common import calc_drawdown, calc_sharp_ratio
from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.instrument.model import Instrument

from statsmodels.tsa.vector_ar.vecm import coint_johansen

tz_local = datetime.datetime.now().astimezone().tzinfo

__SUPPORTED_INSTRUMENTS = [
    # Energy ETFs
    Instrument(symbol='IYE'),
    Instrument(symbol='VDE'),
    Instrument(symbol='XLE'),
    Instrument(symbol='XOP'),
    # Instrument(symbol='XLU'),
    Instrument(symbol='XES'),
    # Uranium ETFs
    Instrument(symbol='URA'),
    Instrument(symbol='SRUUF'),
]


def place_orders_callback():
    try:
        app.place_orders(rebalance_orders)
    except ValueError as err:
        st.toast(f"Got error: {err}")
        st.stop()
    st.balloons()


def save_current_positions_callback():
    try:
        app.save_current_positions()
    except ValueError as err:
        st.toast(f"Got error: {err}")
        st.stop()
    st.balloons()


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

if not input_completed:
    st.stop()

historical_data = app.get_historical_data(
    instruments=chosen_instruments, start_date=datetime.datetime(2015, 1, 1))

# Live trading section
live_trading_container = st.container()
live_trading_container.header("Live trading")
auth_err = app.check_auth()
if auth_err:
    live_trading_container.write(f"Got auth error: {auth_err}")
    live_trading_container.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
else:
    current_portfolio = app.get_current_portfolio()
    cash = current_portfolio.get_cash_balance()
    cash_for_next_positions = cash / 2  # TODO: integrate with Kelly formula later
    positions_df = current_portfolio.get_positions_df()
    positions_df['at'] = positions_df['at'].dt.tz_convert(tz_local)
    # Current positions
    with live_trading_container.container():
        live_trading_container.metric(
            "Cash balance", cash)
        live_trading_container.metric(
            "Cash for next positions", cash_for_next_positions)
        # Pnl history
        pnl = app.calc_pnl_history()
        live_trading_container.expander("PnL history").line_chart(pnl.cumsum())

        live_trading_container.subheader("Current positions")
        _, col2 = live_trading_container.columns([8, 2])
        col2.button("Save positions", on_click=save_current_positions_callback,
                    type='primary', use_container_width=True)
        live_trading_container.table(positions_df)
        # Next positions
        live_trading_container.subheader("Next positions")
        hedge_ratio, spread, spread_std = app.mean_reversion_strategy.calc_trading_inputs(
            price=historical_data.dropna())
        next_hedge_ratio = hedge_ratio.sort_index().iloc[-1]
        next_positions = next_hedge_ratio * cash_for_next_positions / (
            next_hedge_ratio*historical_data.iloc[-1]).abs().sum()
        live_trading_container.table(next_positions)

        rebalance_orders = app.calculate_rebalance_orders(
            next_positions=next_positions)

        # TODO: think about other criteria
        # Maybe sent orders but not filled -> need to block
        is_placeable = rebalance_orders.any() != 0
        live_trading_container.subheader("Orders will be rebalancing")
        _, col2 = st.columns([8, 2])
        if is_placeable:
            col2.button("Place orders", on_click=place_orders_callback,
                        type='primary', use_container_width=True)
        live_trading_container.table(rebalance_orders)


st.divider()

# Backtest section

backtest_container = st.container()
backtest_container.header("Backtest")
# TODO: Setup flow for date range backtest
# date_range = st.date_input("Date range")

start_date, end_date = pd.Timestamp(
    historical_data.index.min()), pd.Timestamp(historical_data.index.max())

in_sample_dataset, out_sample_dataset = train_test_split(
    historical_data.dropna(), test_size=0.3, shuffle=False)

backtest_container.metric(
    "Date range", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}", f"{end_date - start_date}")
backtest_container.line_chart(historical_data)

insample_section, outsample_section = backtest_container.tabs(
    ['IS Summary', 'OS Summary'])
# In sample section
start_date, end_date = pd.Timestamp(
    in_sample_dataset.index.min()), pd.Timestamp(in_sample_dataset.index.max())
insample_section.metric(f"Date Range", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}",
                        f"{end_date - start_date}")


hedge_ratio, spread, spread_std = app.mean_reversion_strategy.calc_trading_inputs(
    price=in_sample_dataset)
ret = app.mean_reversion_strategy.trade(
    in_sample_dataset, hedge_ratio, spread, spread_std)
levered_result = app.mean_reversion_strategy.get_leverage_ratio(ret)
insample_section.write(levered_result)

# Cointegration statistic
insample_section.text("Cointegration statistic")
jres = coint_johansen(in_sample_dataset, det_order=0, k_ar_diff=1)
coint_result = pd.DataFrame(np.concatenate([jres.max_eig_stat.reshape(-1, 1), jres.max_eig_stat_crit_vals], axis=1),
                            columns=['Statistic', 'Crit 90%', 'Crit 95%', 'Crit 99%'])
insample_section.table(coint_result)
halflife = app.mean_reversion_strategy.calc_halflife(
    spread=(hedge_ratio*in_sample_dataset).sum(axis=1))
insample_section.metric("Halflife", halflife)

insample_section.metric("Sharp ratio", round(calc_sharp_ratio(ret), 2))
insample_section.metric("Trades", ret.count())

# Drawdown
drawdown = calc_drawdown(ret)
max_drawdown = drawdown.loc[drawdown['drawdown_deep'].idxmax()] if len(
    drawdown) > 0 else None
max_drawdown_duration = drawdown.loc[drawdown['drawdown_duration'].idxmax()] if len(
    drawdown) > 0 else None
if max_drawdown is not None and max_drawdown_duration is not None:
    col_max_drawdown_deep, col_max_drawdown_duration = insample_section.columns(
        2)
    col_max_drawdown_deep.metric("Max drawdown",
                                 f"-{round(max_drawdown['drawdown_deep']*100, 2)}%")
    col_max_drawdown_duration.metric("Max drawdown duration",
                                     f"{max_drawdown_duration['drawdown_duration']}")
metric_detail = insample_section.expander("Detail")
with metric_detail:
    col1, col2 = metric_detail.columns(2)
    col1.text("Return summary")
    col1.write(ret.describe())
    col2.text("Return")
    col2.write(ret)
    col1, col2 = metric_detail.columns(2)
    with col1:
        col1.text("Max drawdown")
        col1.write(max_drawdown)
    with col2:
        col2.text("Max drawdown duration")
        col2.write(max_drawdown_duration)

aggregrated_by_year = ret.groupby(ret.index.year).agg(
    sharp_ratio=calc_sharp_ratio,
    max_drawdown_deep=lambda x: calc_drawdown(x)['drawdown_deep'].max(),
    max_drawdown_duration=lambda x: calc_drawdown(
        x)['drawdown_duration'].max(),
)
aggregrated_by_year['max_drawdown_duration'] = aggregrated_by_year['max_drawdown_duration'].dt.days
insample_section.table(aggregrated_by_year)

insample_section.line_chart(ret.dropna().cumsum())


# Spread & Std
insample_section.text("Spread & Std")
spread_summary = insample_section.expander("Summary")
with spread_summary:
    col1, col2 = spread_summary.columns(2)
    with col1:
        col1.text("Spread")
        col1.text(spread.describe())
    with col2:
        col2.text("Spread Std")
        col2.text(spread_std.describe())

insample_section.line_chart(pd.DataFrame({
    'spread': spread,
    'std': spread_std,
    '-std': -spread_std
}))

outsample_section.subheader("Out sample test")
