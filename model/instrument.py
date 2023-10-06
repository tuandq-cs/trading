class Instrument:
    symbol: str
    type: str  # TODO: make it enums

    def __init__(self, symbol: str, instrument_type: str) -> None:
        self.symbol = symbol
        self.type = instrument_type

    def __repr__(self) -> str:
        return f'{self.symbol}_{self.type}'
