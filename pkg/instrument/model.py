
from dataclasses import dataclass


@dataclass(repr=False)
class Instrument:
    symbol: str
    type: str = ''  # TODO: make it enums + handle for type
    broker: str = ''
    broker_instrument_id: str = ''

    # def __init__(self, symbol: str, instrument_type: str, broker: str, broker_instrument_id: str) -> None:
    #     self.symbol = symbol
    #     self.type = instrument_type
    #     self.broker = broker
    #     self.broker_instrument_id = broker_instrument_id

    def __repr__(self) -> str:
        return f'{self.symbol}'
