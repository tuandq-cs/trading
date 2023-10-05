
from brokers.interactive_broker import InteractiveBrokers
from portfolio.model import Portfolio


class PortfolioService:
    __interactive_broker: InteractiveBrokers

    def __init__(self, interactive_broker: InteractiveBrokers) -> None:
        self.__interactive_broker = interactive_broker

    def get_current_portfolio(self) -> Portfolio:
        return self.__interactive_broker.get_current_portfolio()
