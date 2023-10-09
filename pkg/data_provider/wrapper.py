
import datetime

import pandas as pd
from constants.error import ERR_DATA_NOT_IN_DISK
from pkg.data_provider.disk import Disk
from pkg.data_provider.twelvedata import TwelveData
from pkg.instrument.model import Instrument


class DataProviderWrapper():
    __twelve: TwelveData
    __disk:  Disk

    def __init__(self) -> None:
        self.__twelve = TwelveData()
        self.__disk = Disk()

    def load_historical_data(self, instrument: Instrument, start_date: datetime.datetime) -> pd.DataFrame:
        try:
            return self.__disk.load_historical_data(instrument=instrument)
        except ValueError as err:
            if err is not ERR_DATA_NOT_IN_DISK:
                raise err
            data = self.__twelve.load_historical_data(
                instrument=instrument, start_date=start_date)
            self.__disk.save_historical_data(
                instrument=instrument, data=data)
            return data
