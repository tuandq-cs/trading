from typing import List
import numpy as np

import pandas as pd
from constants.common import START_HISTORICAL_DATE
from data_provider.wrapper import DataProviderWrapper
from model.instrument import Instrument
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class MeanReversionStrategy:
    __data_provider: DataProviderWrapper
    __LOOK_BACK: int = 46  # TODO: calc this value on training data
    __ENTRY_ZSCORE: float = 2.0
    __EXIT_ZSCORE: float = 0.5

    def __init__(self, data_provider: DataProviderWrapper) -> None:
        self.__data_provider = data_provider

    def __get_hedge_ratio(self, price: pd.DataFrame) -> pd.DataFrame:
        """
            Calculate hedge ratio
        """
        hedge_ratio = pd.DataFrame(
            np.NaN, columns=price.columns, index=price.index)
        for i in range(self.__LOOK_BACK-1, len(price)):
            price_period = price[i-self.__LOOK_BACK+1:i+1]
            jres = coint_johansen(price_period, det_order=0, k_ar_diff=1)
            hedge_ratio.iloc[i] = jres.evec.T[0]
        return hedge_ratio

    def generate_trading_signal(self, instruments: List[Instrument]):
        price_map = {}
        for instrument in instruments:
            historical_data = self.__data_provider.load_historical_data(
                instrument=instrument, start_date=START_HISTORICAL_DATE
            )
            price_map[str(instrument)] = historical_data['close']
        # data_df = pd.DataFrame(historical_data_table)

        price = pd.DataFrame(price_map).sort_index()
        hedge_ratio = self.__get_hedge_ratio(price=price)
        mkt_value = price * hedge_ratio
        spread = mkt_value.sum(axis=1)
        zscore_spread = (spread - spread.rolling(self.__LOOK_BACK).mean()
                         ) / spread.rolling(self.__LOOK_BACK).std()

        long_entry_signal = zscore_spread < -self.__ENTRY_ZSCORE
        long_exit_signal = zscore_spread >= -self.__EXIT_ZSCORE
        long_num_units = pd.Series(np.NaN, index=zscore_spread.index)
        long_num_units.iloc[0] = 0
        long_num_units[long_entry_signal] = 1
        long_num_units[long_exit_signal] = 0
        long_num_units = long_num_units.fillna(method='ffill')

        short_entry_signal = zscore_spread > self.__ENTRY_ZSCORE
        short_exit_signal = zscore_spread <= self.__EXIT_ZSCORE
        short_num_units = pd.Series(np.NaN, index=zscore_spread.index)
        short_num_units.iloc[0] = 0
        short_num_units[short_entry_signal] = -1
        short_num_units[short_exit_signal] = 0
        short_num_units = short_num_units.fillna(method='ffill')

        num_units = long_num_units + short_num_units
        return hedge_ratio.mul(num_units, axis=0)
