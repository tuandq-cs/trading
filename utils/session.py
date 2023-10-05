import streamlit as st
from brokers.interactive_broker import ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY, InteractiveBrokers
from constants.url import INTERACTIVE_BROKER_BASE_URL
from portfolio.service import PortfolioService


@st.cache_resource
def __init_broker() -> InteractiveBrokers:
    return InteractiveBrokers(base_url=INTERACTIVE_BROKER_BASE_URL)


@st.cache_resource
def __init_portfolio_service(broker: InteractiveBrokers) -> PortfolioService:
    return PortfolioService(interactive_broker=broker)


def fetch_session():
    broker = __init_broker()
    portfolio_service = __init_portfolio_service(broker=broker)
    try:
        broker.auth()
    except ValueError as err:
        st.toast(f"Got error: {err}")
        if err is ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY:
            st.write("Please login first")
            st.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
        st.stop()
    st.session_state.broker = broker
    st.session_state.portfolio_service = portfolio_service
