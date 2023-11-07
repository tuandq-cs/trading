
import os
from pathlib import Path
from typing import List

import pandas as pd

from pkg.portfolio.model import PositionDetail


class DiskStorage:
    __position_history_file_path = Path('data/position_history.csv')

    def __init__(self) -> None:
        self.__position_history_file_path.parent.mkdir(
            parents=True, exist_ok=True)

    def save_positions(self, positions: List[PositionDetail]):
        positions_df = pd.DataFrame([
            position.to_dict() for position in positions
        ])
        if len(positions_df) > 0:
            include_header = not os.path.isfile(
                self.__position_history_file_path)
            positions_df.to_csv(
                self.__position_history_file_path, mode='a', header=include_header, index=False)

    def load_positions_history(self):
        return pd.read_csv(self.__position_history_file_path)


class PortfolioRepo:
    __disk_storage: DiskStorage

    def __init__(self) -> None:
        self.__disk_storage = DiskStorage()

    def save_positions(self, positions: List[PositionDetail]):
        self.__disk_storage.save_positions(positions=positions)

    def get_positions_history(self):
        return self.__disk_storage.load_positions_history()
