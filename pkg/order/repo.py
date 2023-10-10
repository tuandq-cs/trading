

import os
from pathlib import Path
from typing import List

import pandas as pd

from pkg.order.model import Order


class DiskStorage:
    __order_request_history_file_path = Path('data/order_request_history.csv')

    def __init__(self) -> None:
        self.__order_request_history_file_path.parent.mkdir(
            parents=True, exist_ok=True)

    def save_order_requests(self, order_requests: List[Order]):
        print(order_requests)
        order_request_df = pd.DataFrame([
            order_request.to_dict() for order_request in order_requests
        ])
        print(order_request_df)
        if len(order_request_df) > 0:
            include_header = not os.path.isfile(
                self.__order_request_history_file_path)
            order_request_df.to_csv(
                self.__order_request_history_file_path, mode='a', header=include_header, index=False)


class OrderRepo:
    __disk_storage: DiskStorage

    def __init__(self) -> None:
        self.__disk_storage = DiskStorage()

    def save_order_requests(self, order_requests: List[Order]):
        self.__disk_storage.save_order_requests(order_requests)
