

import datetime
import io
import pandas as pd
import requests

from pkg.instrument.model import Instrument


class TwelveData():
    __base_url: str = 'https://api.twelvedata.com'
    __api_key: str = '70921edd203f44aab576fefe41befc92'

    def load_historical_data(self, instrument: Instrument, start_date: datetime.datetime) -> pd.DataFrame:
        url = f"{self.__base_url}/time_series" \
            f"?apikey={self.__api_key}" \
            f"&symbol={instrument.symbol}" \
            f"&interval=1day&type={instrument.type}&format=CSV" \
            f"&start_date={start_date.strftime('%Y-%m-%d')}"
        response = requests.get(url, timeout=60)
        return pd.read_csv(io.StringIO(
            response.content.decode('utf-8')), delimiter=";").set_index("datetime")
