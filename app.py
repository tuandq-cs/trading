from typing import List
import pandas as pd
import streamlit as st
from constants.error import err_not_support_instrument

from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.brokers.interactive_broker import InteractiveBrokers
from pkg.data_provider.wrapper import DataProviderWrapper
from pkg.order.model import Order, OrderSide
from pkg.order.service import OrderService
from pkg.portfolio.model import Portfolio
from pkg.portfolio.service import PortfolioService
from pkg.strategies.mean_reversion import MeanReversionStrategy


class App():
    broker: InteractiveBrokers
    portfolio_service: PortfolioService
    order_service: OrderService
    data_provider: DataProviderWrapper
    mean_reversion_strategy: MeanReversionStrategy

    def __init__(self) -> None:
        self.broker = InteractiveBrokers(base_url=INTERACTIVE_BROKER_BASE_URL)
        self.portfolio_service = PortfolioService(
            interactive_broker=self.broker)
        self.order_service = OrderService(broker=self.broker)
        self.data_provider = DataProviderWrapper()
        self.mean_reversion_strategy = MeanReversionStrategy(
            data_provider=self.data_provider)

    def fetch_session(self):
        try:
            self.broker.auth()
        except ValueError as err:
            st.toast(f"Got error: {err}")
            st.write("Please login first")
            st.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
            st.stop()

    def get_instrument_map(self) -> dict:
        instruments = self.broker.get_supported_instruments()
        instrument_map = {}
        for instrument in instruments:
            instrument_map[str(instrument)] = instrument
        return instrument_map

    def get_current_portfolio(self) -> Portfolio:
        return self.portfolio_service.get_current_portfolio()

    def calculate_rebalance_orders(self, next_positions: pd.Series) -> pd.Series:
        current_portfolio = self.get_current_portfolio()
        positions_df = current_portfolio.get_positions_df()
        current_positions = positions_df['position']
        rebalance_orders = next_positions.subtract(
            current_positions, fill_value=0)
        rebalance_orders = rebalance_orders.loc[next_positions.index]
        return rebalance_orders[rebalance_orders != 0]

    def place_orders(self, orders_series: pd.Series):
        orders: List[Order] = []
        instrument_map = self.get_instrument_map()
        for instrument_key in orders_series.index:
            instrument = instrument_map.get(instrument_key)
            if not instrument:
                raise err_not_support_instrument(instrument_key)
            qty = round(orders_series.loc[instrument_key], 4)
            side = OrderSide.BUY if qty >= 0 else OrderSide.SELL
            orders.append(Order(instrument=instrument, quantity=abs(qty),
                          side=side))
        self.order_service.place_orders(orders=orders)

    def save_current_positions(self):
        self.portfolio_service.save_current_portfolio()

    def get_orders_history(self) -> pd.DataFrame:
        return self.order_service.get_orders_history_df()


@st.cache_resource
def init_app() -> App:
    return App()
