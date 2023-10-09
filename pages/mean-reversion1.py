from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from app import init_app
from pkg.instrument.model import Instrument


app = init_app()
app.fetch_session()
st.title("Mean reversion")

current_portfolio = app.portfolio_service.get_current_portfolio()
positions_df = current_portfolio.get_positions_df().set_index('instrument')
st.header("Current positions")
st.table(positions_df)


def supported_instruments() -> List[Instrument]:
    return [
        Instrument('IYE', 'etf'),
        Instrument('VDE', 'etf')
    ]


def place_orders_callback():
    # TODO: implement here
    st.balloons()


st.header("Choose instruments")
chosen_instruments = st.multiselect("Instruments", supported_instruments())

if len(chosen_instruments) >= 2:
    zscore_spread, generated_positions = app.mean_reversion_strategy.generate_trading_signal(
        chosen_instruments)
    next_positions = generated_positions.sort_index().iloc[-1]
    st.subheader("Next positions")
    st.table(next_positions)

    current_positions = positions_df['position']
    rebalance_positions = pd.Series(np.NaN, index=next_positions.index)
    rebalance_positions = next_positions.subtract(
        current_positions, fill_value=0)
    rebalance_positions = rebalance_positions.loc[next_positions.index]
    rebalance_positions = rebalance_positions[rebalance_positions != 0]

    placed_orders_of_current_date = False
    is_placeable = not placed_orders_of_current_date and len(
        rebalance_positions) > 0
    if is_placeable:
        st.button("Place orders", on_click=place_orders_callback)
        st.subheader("Rebalance positions")
        st.table(rebalance_positions)

    # Display rebalancing positions
    st.subheader("Zscore Spread")
    st.line_chart(data=zscore_spread.sort_index())

# TODO: display backtest result here
