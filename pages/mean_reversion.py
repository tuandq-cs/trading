from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from app import init_app
from pkg.instrument.model import Instrument


app = init_app()
app.fetch_session()


def place_orders_callback():
    try:
        app.place_orders(rebalance_orders)
    except ValueError as err:
        st.toast(f"Got error: {err}")
        st.stop()
    st.balloons()


st.title("Mean reversion")
current_portfolio = app.get_current_portfolio()
positions_df = current_portfolio.get_positions_df().set_index('instrument')
st.header("Current positions")
st.table(positions_df)

orders_history = app.get_orders_history()
orders_history['created_at'] = orders_history['created_at'].dt.tz_convert(
    tz='US/Eastern')

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
    if is_placeable:
        st.button("Place orders", on_click=place_orders_callback)
        st.subheader("Orders will be rebalancing")
        st.table(rebalance_orders)

    # Display rebalancing positions
    st.subheader("Zscore Spread")
    st.line_chart(data=zscore_spread.sort_index())

# Display order requests of chosen instruments historically
st.subheader("Order requests history")
st.table(orders_history)
# TODO: display backtest result here
