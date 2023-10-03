import streamlit as st
import requests
import io
import os
import pandas as pd
import datetime
import numpy as np
import streamlit.components.v1 as components
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from utils.session import fetch_session

fetch_session()
if 'authenticated' not in st.session_state:
    st.write("Please login first")
    st.link_button("Login", "https://localhost:5000")
    st.stop()

START_HISTORICAL_DATE = datetime.datetime(2015, 1, 1)
LOOK_BACK = 46
ENTRY_ZSCORE = 2
EXIT_ZSCORE = 0

st.title("Position summary")


@st.cache_data
def load_historical_data():
    """
        Load data
    """
    symbols = ['IYE', 'VDE']
    now = datetime.datetime.now()
    data_dir = f"data/{now.strftime('%Y%m%d')}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    raw_data = {}
    for symbol in symbols:
        file_name = f"{data_dir}/{symbol}.csv"
        try:
            data = pd.read_csv(file_name, delimiter=",")
        except IOError as err:
            print(err)
            url = "https://api.twelvedata.com/time_series" \
                "?apikey=70921edd203f44aab576fefe41befc92" \
                f"&symbol={symbol}" \
                "&interval=1day&type=etf&format=CSV" \
                f"&start_date={START_HISTORICAL_DATE.strftime('%Y-%m-%d')}"
            response = requests.get(url, timeout=60)
            data = pd.read_csv(io.StringIO(
                response.content.decode('utf-8')), delimiter=";")
            data.to_csv(file_name, mode='w', index=False)
        data = data.set_index("datetime")
        raw_data[symbol] = data
    return raw_data


@st.cache_data
def get_hedge_ratio(close_price):
    """
        Calculate hedge ratio
    """
    hedge_ratio = pd.DataFrame(
        np.NaN, columns=close_price.columns, index=close_price.index)
    for i in range(LOOK_BACK-1, len(close_price)):
        price_period = close_price[i-LOOK_BACK+1:i+1]
        jres = coint_johansen(price_period, det_order=0, k_ar_diff=1)
        hedge_ratio.iloc[i] = jres.evec.T[0]
    return hedge_ratio


# Interactive Brokers place order's API
def place_orders(orders: pd.DataFrame):
    # TODO: handle here
    if len(orders) == 0:
        return
    # Fetch conid from api
    # TODO: put to func
    conid_map = {}
    for symbol in orders.index:
        if conid_map.get(symbol):
            continue
        response = requests.post(
            "https://localhost:5000/v1/api/iserver/secdef/search", timeout=60,
            verify=False,
            json={
                'symbol': symbol,
                'secType': 'STK'
            })
        response_body = response.json()
        if len(response_body) == 0:
            # TODO: handle error
            st.toast('Fuck! Check please')
            return
        conid_map[symbol] = int(response_body[0]['conid'])
    # Prepare request body
    order_requests = []
    parent_order_id = None
    for symbol in orders.index:
        qty = orders.loc[symbol]['position']
        order_req = {
            'conid': conid_map[symbol],
            'orderType': 'MKT',  # TODO: handle for other types
            'side': 'BUY' if qty > 0 else 'SELL',
            'quantity': abs(qty.round(4)),
            'tif': "DAY"  # TODO: handle for other tif
        }
        if parent_order_id is None:
            parent_order_id = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            order_req['cOID'] = parent_order_id
        else:
            order_req['parentId'] = parent_order_id
        order_requests.append(order_req)
    # Call API place orders
    url = f"https://localhost:5000/v1/api/iserver/account/{st.session_state['selected_account']}/orders"
    response = requests.post(url=url, timeout=60, verify=False,
                             json={
                                 'orders': order_requests
                             })
    response_body = response.json()
    st.write(order_requests)
    st.write(response)
    st.write(response_body)


def place_orders_callback():
    abstract_orders = rebalance_positions
    st.write(abstract_orders)
    place_orders(abstract_orders)
    st.balloons()


historical_data_map = load_historical_data()
# Extract data
close_price = {}
for symbol, symbol_data in historical_data_map.items():
    close_price[symbol] = symbol_data["close"]
close_price = pd.DataFrame(close_price).sort_index()
hedge_ratio = get_hedge_ratio(close_price=close_price)
mkt_value = hedge_ratio * close_price
spread = mkt_value.sum(axis=1)
zscore_spread = (spread - spread.rolling(LOOK_BACK).mean()) / \
    spread.rolling(LOOK_BACK).std()

# Current position
current_positions = pd.DataFrame(
    [-63.5, 115], index=hedge_ratio.columns, columns=['position'])

# Signal
long_entry_signal = zscore_spread < -ENTRY_ZSCORE
short_entry_signal = zscore_spread > ENTRY_ZSCORE
long_exit_signal = zscore_spread >= EXIT_ZSCORE
short_exit_signal = zscore_spread <= EXIT_ZSCORE

entry_signal = long_entry_signal + short_entry_signal
exit_signal = long_exit_signal + short_exit_signal
last_entry_signal = entry_signal.iloc[-1]
last_exit_signal = exit_signal.iloc[-1]

st.header("Current positions")
current_positions


st.header("Next positions")
next_positions = current_positions
last_exit_signal = 0
last_entry_signal = -2
if last_exit_signal:
    st.checkbox('Exit signal', value=True, disabled=True)
    next_positions = pd.DataFrame(
        [0] * len(current_positions), index=current_positions.index, columns=current_positions.columns)
if last_entry_signal:
    st.checkbox('Entry signal', value=True, disabled=True)
    next_positions = pd.DataFrame(
        (hedge_ratio*last_entry_signal).iloc[-1].values, index=hedge_ratio.columns, columns=['position'])

rebalance_positions = current_positions - next_positions
if (rebalance_positions['position'] != 0).any():
    st.subheader("Rebalance positions")
    st.write(rebalance_positions)
    st.button("Place orders", on_click=place_orders_callback)

# History spread
st.subheader("Spread history (zscore)")
st.line_chart(zscore_spread.dropna().tail(20))
