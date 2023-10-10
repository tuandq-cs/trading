

from typing import List
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
        print(order_requests)
        self.__repo.save_order_requests(order_requests=order_requests)
