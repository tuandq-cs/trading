import streamlit as st

from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.brokers.interactive_broker import ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY, InteractiveBrokers
from pkg.data_provider.wrapper import DataProviderWrapper
from pkg.portfolio.service import PortfolioService
from pkg.strategies.mean_reversion import MeanReversionStrategy


class App():
    broker: InteractiveBrokers
    portfolio_service: PortfolioService
    data_provider: DataProviderWrapper
    mean_reversion_strategy: MeanReversionStrategy

    def __init__(self) -> None:
        self.broker = InteractiveBrokers(base_url=INTERACTIVE_BROKER_BASE_URL)
        self.portfolio_service = PortfolioService(
            interactive_broker=self.broker)
        self.data_provider = DataProviderWrapper()
        self.mean_reversion_strategy = MeanReversionStrategy(
            data_provider=self.data_provider)

    def fetch_session(self):
        try:
            self.broker.auth()
        except ValueError as err:
            st.toast(f"Got error: {err}")
            if err is ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY:
                st.write("Please login first")
                st.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
            st.stop()


@st.cache_resource
def init_app() -> App:
    return App()
