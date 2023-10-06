import streamlit as st
from app import init_app

app = init_app()
app.fetch_session()  # hawjwn798
st.write(f"Welcome {app.broker.get_selected_account()}")
