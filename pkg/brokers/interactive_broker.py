from dataclasses import dataclass
import datetime
import json
from typing import List
import requests
from pkg.instrument.model import Instrument
from pkg.order.model import Order, OrderSide

from pkg.portfolio.model import Portfolio, PositionDetail

ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY = ValueError(
    'ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY')
ERR_NOT_FOUND_SELECTED_ACCOUNT = ValueError('ERR_NOT_FOUND_SELECTED_ACCOUNT')


def err_unauthenticated(response_body):
    return ValueError(
        'ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY', response_body)


@dataclass
class InstrumentConfig:
    symbol: str
    type: str
    conid: str  # Contract Identifier


class InteractiveBrokers:
    __name: str = 'InteractiveBrokers'
    __base_url: str
    __selected_account: str
    __PREDEFINED_INSTRUMENT_CONFIGS = [
        InstrumentConfig(symbol='IYE', type='ETF', conid='10190340'),
        InstrumentConfig(symbol='VDE', type='ETF', conid='27684036'),
    ]
    __timeout_second: int = 60
    __supported_instrument_map: dict[str, Instrument] = {}

    def __init__(self, base_url: str) -> None:
        self.__base_url = base_url

    def auth(self) -> None:
        try:
            response = requests.post(
                f'{self.__base_url}/v1/api/iserver/auth/status', timeout=self.__timeout_second, verify=False)
            if response.status_code != 200:
                raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
            response_body = response.json()
            print(response_body)
            if not response_body.get('authenticated'):
                raise err_unauthenticated(response_body=response_body)
            response = requests.get(
                f'{self.__base_url}/v1/api/iserver/accounts', timeout=self.__timeout_second, verify=False)
            selected_account = response.json().get('selectedAccount')
            if not selected_account:
                raise ERR_NOT_FOUND_SELECTED_ACCOUNT
            self.__selected_account = selected_account
        except requests.exceptions.RequestException:
            raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY

    def get_selected_account(self) -> str:
        return self.__selected_account

    def __fetch_supported_instruments(self):
        supported_instrument_map: dict[str, Instrument] = {}
        for instrument_config in self.__PREDEFINED_INSTRUMENT_CONFIGS:
            try:
                response = requests.post(
                    f'{self.__base_url}/v1/api/iserver/secdef/search', timeout=self.__timeout_second, verify=False,
                    json={
                        'symbol': instrument_config.symbol
                    }
                )
                response_body = response.json()
                if response.status_code != 200:
                    raise ValueError(
                        f"Status code: {response.status_code}, Message: {response_body}")
                if type(response_body) is not list or len(response_body) == 0 or not response_body[0].get('conid'):
                    raise ValueError("INVALID RESPONSE", response_body)
                if response_body[0].get('conid') != instrument_config.conid:
                    raise ValueError(
                        f"Unmatch conid of instrument {instrument_config.symbol}")
                supported_instrument_map[instrument_config.conid] = Instrument(
                    symbol=instrument_config.symbol, instrument_type=instrument_config.type, broker=self.__name, broker_instrument_id=instrument_config.conid)
            except requests.exceptions.RequestException:
                raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
        self.__supported_instrument_map = supported_instrument_map

    def get_supported_instruments(self) -> List[Instrument]:
        if len(self.__supported_instrument_map) == 0:
            self.__fetch_supported_instruments()
        return list(self.__supported_instrument_map.values())

    def get_current_portfolio(self) -> Portfolio:
        page_id = 0
        response = requests.get(
            f"{self.__base_url}/v1/api/portfolio/{self.__selected_account}/positions/{page_id}", timeout=self.__timeout_second,
            verify=False)
        if response.status_code != 200:
            raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY
        current_positions: list[PositionDetail] = []
        for item in response.json():
            # TODO: handle instrument_type
            instrument = Instrument(
                symbol=item['contractDesc'], instrument_type='', broker=self.__name, broker_instrument_id=item['conid'])
            if self.__supported_instrument_map.get(item['conid']):
                instrument = self.__supported_instrument_map[item['conid']]
            current_positions.append(PositionDetail(
                instrument=instrument, position=item['position']))
        return Portfolio(positions=current_positions)

    def place_orders(self, orders: List[Order]) -> List[Order]:
        now = datetime.datetime.now()
        for order in orders:
            try:
                order.request_order_id = f'{now.strftime("%Y%m%d%H%M%S")}-{order.instrument.symbol}'
                request_payload = {
                    'orders': [
                        {
                            'conid': int(order.instrument.broker_instrument_id),
                            'cOID': order.request_order_id,
                            'orderType': 'MKT',  # TODO: handle orderType here
                            'side': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                            'quantity': order.quantity,
                            'tif': 'DAY'  # TODO: handle tif here
                        }
                    ]
                }
                order.status = 'SENDING_REQUEST'
                order.request_payload = json.dumps(request_payload)
                order.created_at = now
                response = requests.post(
                    f'{self.__base_url}/v1/api/iserver/account/{self.get_selected_account()}/orders', timeout=self.__timeout_second, verify=False,
                    json=request_payload
                )
                order.response_payload = response.text
                response_body = response.json()
                if response.status_code != 200:
                    order.status = 'SENT_REQUEST_FAILED'
                    raise ValueError(
                        f"Status code: {response.status_code}, Message: {response_body}")
                if not isinstance(response_body, list) and len(response_body) == 0:
                    order.status = 'FAILED_TO_PARSE_RESPONSE_BODY'
                    raise ValueError("INVALID_RESPONSE_BODY")
                order.status = 'SENT_REQUEST_SUCCESSFUL'
                order.broker_order_id = response_body[0].get('order_id')
            except requests.exceptions.RequestException as err:
                raise ERR_UNAUTHENTICATED_CLIENT_PORTAL_GATEWAY from err
        return orders
