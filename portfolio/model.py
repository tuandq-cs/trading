
from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class PositionDetail:
    broker: str
    instrument: str
    broker_instrument_id: str
    position: int


class Portfolio:
    __positions: List[PositionDetail]

    def __init__(self, positions: List[PositionDetail]) -> None:
        self.__positions = positions

    def get_positions_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.__positions)
