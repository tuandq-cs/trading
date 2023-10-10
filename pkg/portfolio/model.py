
from dataclasses import dataclass
from typing import List
import pandas as pd

from pkg.instrument.model import Instrument


@dataclass
class PositionDetail:
    instrument: Instrument
    position: int


class Portfolio:
    __positions: List[PositionDetail]

    def __init__(self, positions: List[PositionDetail]) -> None:
        self.__positions = positions

    def get_positions_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'instrument': str(position.instrument),
                'position': position.position,
            } for position in self.__positions])
