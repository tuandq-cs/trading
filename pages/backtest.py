

import streamlit as st
from app import init_app


app = init_app()
# app.fetch_session()
instrument_map = app.get_instrument_map_for_backtest()
st.header("Choose instruments")
chosen_instruments = st.multiselect("Instruments", instrument_map.values())
if len(chosen_instruments) >= 2:
    app.mean_reversion_strategy.backtest(chosen_instruments)
