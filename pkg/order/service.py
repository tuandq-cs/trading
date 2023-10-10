

from typing import List

import pandas as pd
from pkg.brokers.interactive_broker import InteractiveBrokers
from pkg.order.model import Order
from pkg.order.repo import OrderRepo


class OrderService():
    __broker: InteractiveBrokers
    __repo: OrderRepo

    def __init__(self, broker: InteractiveBrokers) -> None:
        self.__broker = broker
        self.__repo = OrderRepo()

    def place_orders(self, orders: List[Order]):
        order_requests = self.__broker.place_orders(orders=orders)
        self.__repo.save_order_requests(order_requests=order_requests)

    def get_orders_history_df(self) -> pd.DataFrame:
        raw_orders_requests_history = self.__repo.get_orders_requests_history()
        columns = [
            'broker_order_id', 'request_order_id',
            'symbol', 'broker_instrument_id',
            'order_type', 'side', 'quantity', 'created_at'
        ]
        orders_request_history = raw_orders_requests_history[columns]
        orders_request_history.loc[:, 'created_at'] = pd.to_datetime(
            orders_request_history['created_at'], unit='s', utc=True)
        return orders_request_history
