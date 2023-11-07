from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from constants.common import START_HISTORICAL_DATE
from constants.error import err_not_support_instrument

from constants.url import INTERACTIVE_BROKER_BASE_URL
from pkg.brokers.interactive_broker import InteractiveBrokers
from pkg.data_provider.wrapper import DataProviderWrapper
from pkg.instrument.model import Instrument
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

    __SUPPORTED_INSTRUMENTS = [
        Instrument(symbol='IYE'),
        Instrument(symbol='VDE'),
        Instrument(symbol='XLE'),
        Instrument(symbol='FENY'),
        Instrument(symbol='XES')
    ]

    def __init__(self) -> None:
        self.broker = InteractiveBrokers(base_url=INTERACTIVE_BROKER_BASE_URL)
        self.portfolio_service = PortfolioService(
            interactive_broker=self.broker)
        self.order_service = OrderService(broker=self.broker)
        self.data_provider = DataProviderWrapper()
        self.mean_reversion_strategy = MeanReversionStrategy(
            data_provider=self.data_provider)

    def check_auth(self):
        try:
            self.broker.auth()
        except ValueError as err:
            return err

    def fetch_session(self):
        try:
            self.broker.auth()
        except ValueError as err:
            st.toast(f"Got error: {err}")
            st.write("Please login first")
            st.link_button("Login", INTERACTIVE_BROKER_BASE_URL)
            st.stop()

    def get_instrument_map_for_backtest(self) -> dict:
        instrument_map = {}
        for instrument in self.__SUPPORTED_INSTRUMENTS:
            instrument_map[str(instrument)] = instrument
        return instrument_map

    def get_instrument_map(self) -> dict:
        instruments = self.broker.get_supported_instruments()
        instrument_map = {}
        for instrument in instruments:
            instrument_map[str(instrument)] = instrument
        return instrument_map

    def get_historical_data(self, instruments: List[Instrument], start_date=START_HISTORICAL_DATE) -> pd.DataFrame:
        df_list: list[pd.DataFrame] = []
        for instrument in instruments:
            historical_data = self.data_provider.load_historical_data(
                instrument=instrument, start_date=start_date
            )
            df_list.append(pd.DataFrame({
                str(instrument): historical_data[~historical_data.index.duplicated()]['close']
            }))
        return pd.concat(df_list, axis=1).sort_index()

    def split_historical_data(self, historical_data: pd.DataFrame, num_folds: int):
        tscv = TimeSeriesSplit(n_splits=num_folds)
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for _, (train_indices, valid_indices) in enumerate(tscv.split(historical_data)):
            train_data, valid_data = historical_data.iloc[
                train_indices], historical_data.iloc[valid_indices]
            folds.append((train_data, valid_data))
        return folds

    def get_current_portfolio(self) -> Portfolio:
        return self.portfolio_service.get_current_portfolio()

    def calculate_rebalance_orders(self, next_positions: pd.Series) -> pd.Series:
        current_portfolio = self.get_current_portfolio()
        positions_df = current_portfolio.get_positions_df()
        current_positions = positions_df['position']
        rebalance_orders = next_positions.subtract(
            current_positions, fill_value=0)
        rebalance_orders = rebalance_orders.loc[next_positions.index]
        return rebalance_orders[rebalance_orders != 0].transform(lambda x: round(x, 4))

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

    def calc_pnl_history(self):
        positions_history = self.portfolio_service.get_positions_history()
        positions_history['date'] = pd.to_datetime(
            positions_history['at'], unit='s', utc=True).dt.date
        positions_history = positions_history.pivot(
            index='date', columns='symbol', values='position')
        price = self.get_historical_data(
            instruments=[Instrument(symbol=x) for x in positions_history.columns])
        pnl = (price - price.shift(1)) / price.shift(1) * positions_history
        pnl = pnl.dropna().sum(axis=1)
        return pnl

# @st.cache_resource


def init_app() -> App:
    return App()
