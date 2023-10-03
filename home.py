import streamlit as st
import requests

from utils.session import fetch_session

fetch_session()
if 'authenticated' not in st.session_state:
    st.write("Please login first")
    st.link_button("Login", "https://localhost:5000")
    st.stop()
st.write(f"Welcome {st.session_state['selected_account']}")
