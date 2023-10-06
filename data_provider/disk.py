

import datetime
import os

import pandas as pd

from constants.error import ERR_DATA_NOT_IN_DISK
from model.instrument import Instrument


class Disk:
    __base_dir: str = 'data'

    def __get_data_dir(self):
        data_dir = f"{self.__base_dir}/{datetime.datetime.now().strftime('%Y%m%d')}"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        return data_dir

    def load_historical_data(self, instrument: Instrument) -> pd.DataFrame:
        file_name = f"{self.__get_data_dir()}/{instrument}.csv"
        if not os.path.exists(file_name):
            raise ERR_DATA_NOT_IN_DISK
        return pd.read_csv(file_name, delimiter=",").set_index("datetime")

    def save_historical_data(self, instrument: Instrument, data: pd.DataFrame):
        file_name = f"{self.__get_data_dir()}/{instrument}.csv"
        data.to_csv(file_name, mode='w', index=True)
