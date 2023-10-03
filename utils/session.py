import streamlit as st
import requests


def fetch_session():
    st.session_state['authenticated'] = False
    try:
        # TODO:  handle more for not establish localhost 5000
        response = requests.post(
            'https://localhost:5000/v1/api/iserver/auth/status', timeout=60, verify=False)
        if response.status_code != 200:
            return
        response_body = response.json()
        if not response_body.get('authenticated'):
            return
        st.session_state['authenticated'] = True
        if 'selected_account' not in st.session_state:
            response = requests.get(
                'https://localhost:5000/v1/api/iserver/accounts', timeout=60, verify=False)
            response_body = response.json()
            st.session_state['selected_account'] = response_body['selectedAccount']
    except Exception as err:
        st.toast(err)
