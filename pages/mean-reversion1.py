from typing import List
import streamlit as st
from app import init_app
from constants.common import START_HISTORICAL_DATE
from model.instrument import Instrument


app = init_app()
app.fetch_session()
st.title("Mean reversion")

current_portfolio = app.portfolio_service.get_current_portfolio()
positions_df = current_portfolio.get_positions_df()
st.header("Current positions")
positions_df


def supported_instruments() -> List[Instrument]:
    return [
        Instrument('IYE', 'etf'),
        Instrument('VDE', 'etf')
    ]


st.subheader("Choose instruments")
chosen_instruments = st.multiselect("Instruments", supported_instruments())

if len(chosen_instruments) >= 2:
    data = app.mean_reversion_strategy.generate_trading_signal(
        chosen_instruments)
    data
    st.line_chart(data=data)

# TODO: display backtest result here

# Load bucket data of all chosen instruments

# Display any current entry/exit signal
# Place order here
# Display historical market value of the bucket
