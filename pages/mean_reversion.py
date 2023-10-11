import datetime
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from app import init_app
from pkg.instrument.model import Instrument

tz_local = datetime.datetime.now().astimezone().tzinfo

app = init_app()
app.fetch_session()


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


st.title("Mean Reversion Strategy")
st.divider()

current_portfolio = app.get_current_portfolio()
positions_df = current_portfolio.get_positions_df()
positions_df['at'] = positions_df['at'].dt.tz_convert(tz_local)
with st.container():
    st.header("Current positions")
    col1, col2 = st.columns([8, 2])
    col2.button("Save positions", on_click=save_current_positions_callback,
                type='primary', use_container_width=True)
    st.table(positions_df)


orders_history = app.get_orders_history()
orders_history['created_at'] = orders_history['created_at'].dt.tz_convert(
    tz='US/Eastern')

st.divider()

instrument_map = app.get_instrument_map()
st.header("Choose instruments")
chosen_instruments = st.multiselect("Instruments", instrument_map.values())

if len(chosen_instruments) >= 2:
    zscore_spread, generated_positions = app.mean_reversion_strategy.generate_trading_signal(
        chosen_instruments)
    next_positions = generated_positions.sort_index().iloc[-1]
    st.subheader("Next positions")
    st.table(next_positions)

    rebalance_orders = app.calculate_rebalance_orders(
        next_positions=next_positions)

    # TODO: think about other criteria
    # Maybe sent orders but not filled -> need to block
    is_placeable = rebalance_orders.any() != 0
    with st.container():
        st.subheader("Orders will be rebalancing")
        col1, col2 = st.columns([8, 2])
        if is_placeable:
            col2.button("Place orders", on_click=place_orders_callback,
                        type='primary', use_container_width=True)
        st.table(rebalance_orders)

    # Display historical zscore spread
    st.subheader("Zscore Spread")
    zscore_spread = zscore_spread.sort_index()
    date_time_filters = st.date_input(
        "Date range",
        [zscore_spread.index.max() - datetime.timedelta(days=90),
         zscore_spread.index.max()],
        min_value=zscore_spread.index.min(), max_value=zscore_spread.index.max())
    filtered_zscore_spread = zscore_spread

    if isinstance(date_time_filters, tuple):
        if len(date_time_filters) == 1:
            filtered_zscore_spread = zscore_spread[date_time_filters[0]:]
        if len(date_time_filters) == 2:
            filtered_zscore_spread = zscore_spread[date_time_filters[0]:date_time_filters[1]]
    st.line_chart(data=filtered_zscore_spread)

# Display order requests of chosen instruments historically
st.divider()
st.subheader("Order requests history")
st.table(orders_history)
# TODO: display backtest result here
