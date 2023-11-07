
import datetime
from typing import List
import pandas as pd

from pkg.instrument.model import Instrument


class PositionDetail:
    instrument: Instrument
    position: float
    at: datetime.datetime

    def __init__(self, instrument: Instrument, position: float, at: datetime.datetime = datetime.datetime.now().astimezone()):
        self.instrument = instrument
        self.position = position
        self.at = at

    def to_dict(self):
        return {
            'broker': self.instrument.broker,
            'symbol': self.instrument.symbol,
            'broker_instrument_id': self.instrument.broker_instrument_id,
            'position': self.position,
            'at': self.at.timestamp()
        }


class Portfolio:
    __cash_balance: float
    __positions: List[PositionDetail]

    def __init__(self, cash_balance: float, positions: List[PositionDetail]) -> None:
        self.__cash_balance = cash_balance
        self.__positions = positions

    def get_positions(self) -> List[PositionDetail]:
        return self.__positions

    def get_cash_balance(self) -> float:
        return self.__cash_balance

    def get_positions_df(self) -> pd.DataFrame:
        positions_df = pd.DataFrame([
            position.to_dict() for position in self.__positions])
        positions_df.loc[:, 'at'] = pd.to_datetime(
            positions_df['at'], unit='s', utc=True)
        return positions_df.set_index('symbol')
