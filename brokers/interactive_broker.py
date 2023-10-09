import requests
from model.instrument import Instrument
from portfolio.model import Portfolio, PositionDetail

ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY = ValueError(
    'ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY')
ERR_NOT_FOUND_SELECTED_ACCOUNT = ValueError('ERR_NOT_FOUND_SELECTED_ACCOUNT')


class InteractiveBrokers:
    __name: str = 'InteractiveBrokers'
    __base_url: str
    __selected_account: str

    def __init__(self, base_url: str) -> None:
        self.__base_url = base_url

    def auth(self) -> None:
        try:
            response = requests.post(
                f'{self.__base_url}/v1/api/iserver/auth/status', timeout=60, verify=False)
            if response.status_code != 200:
                raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
            response_body = response.json()
            if not response_body.get('authenticated'):
                raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
            response = requests.get(
                f'{self.__base_url}/v1/api/iserver/accounts', timeout=60, verify=False)
            print(response)
            selected_account = response.json().get('selectedAccount')
            if not selected_account:
                raise ERR_NOT_FOUND_SELECTED_ACCOUNT
            self.__selected_account = selected_account
        except requests.exceptions.RequestException:
            raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY

    def get_selected_account(self) -> str:
        return self.__selected_account

    def get_current_portfolio(self) -> Portfolio:
        page_id = 0
        response = requests.get(
            f"{self.__base_url}/v1/api/portfolio/{self.__selected_account}/positions/{page_id}", timeout=60,
            verify=False)
        if response.status_code != 200:
            raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
        current_positions: list[PositionDetail] = []
        for item in response.json():
            # TODO: handle instrument_type
            current_positions.append(PositionDetail(
                broker=self.__name, instrument=Instrument(symbol=item['contractDesc'], instrument_type='etf'), broker_instrument_id=item['conid'], position=item['position']))
        return Portfolio(positions=current_positions)
