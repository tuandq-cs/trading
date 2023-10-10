
from dataclasses import dataclass
import datetime
from enum import Enum

from pkg.instrument.model import Instrument


class OrderSide(Enum):
    BUY = 1
    SELL = 2


class OrderType(Enum):
    Market = 1


@dataclass(init=False)
class Order:
    instrument: Instrument
    quantity: int
    side: OrderSide
    type: OrderType = OrderType.Market
    broker_order_id: str
    request_order_id: str
    status: str  # TODO: consider enums here
    request_payload: str
    created_at: datetime.datetime

    def __init__(self, instrument: Instrument, quantity: int, side: OrderSide) -> None:
        self.instrument = instrument
        self.quantity = quantity
        self.side = side

    def to_dict(self):
        return {
            'broker': self.instrument.broker,
            'broker_order_id': self.broker_order_id,
            'request_order_id': self.request_order_id,
            'symbol': self.instrument.symbol,
            'broker_instrument_id': self.instrument.broker_instrument_id,
            'order_type': self.type.name,
            'side': self.side.name,
            'quantity': self.quantity,
            'created_at': self.created_at.timestamp(),
            'request_payload': self.request_payload
        }
