
from pkg.brokers.interactive_broker import InteractiveBrokers
from pkg.portfolio.model import Portfolio
from pkg.portfolio.repo import PortfolioRepo


class PortfolioService:
    __interactive_broker: InteractiveBrokers
    __repo: PortfolioRepo

    def __init__(self, interactive_broker: InteractiveBrokers) -> None:
        self.__interactive_broker = interactive_broker
        self.__repo = PortfolioRepo()

    def get_current_portfolio(self) -> Portfolio:
        return self.__interactive_broker.get_current_portfolio()

    def save_current_portfolio(self):
        current_portfolio = self.__interactive_broker.get_current_portfolio()
        # Save current positions
        self.__repo.save_positions(current_portfolio.get_positions())
        # TODO: Save other things (cash, margin, ...)

    def get_positions_history(self):
        return self.__repo.get_positions_history()
