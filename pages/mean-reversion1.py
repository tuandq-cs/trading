import streamlit as st
from app import init_app


app = init_app()
app.fetch_session()
st.title("Portfolio summary")

current_portfolio = app.portfolio_service.get_current_portfolio()
positions_df = current_portfolio.get_positions_df()
positions_df
