# app.py
import streamlit as st
from tools.data_management import show as show_data_management
from tools.model_prediction import show as show_model_prediction

# One config — this runs once
st.set_page_config(page_title="JetRacer Tools", layout="wide", initial_sidebar_state="expanded")

# One sidebar in the entire app
st.sidebar.title("JetRacer Toolkit")
page = st.sidebar.radio("Navigation", ["Data Management", "Model Prediction"])

if page == "Data Management":
    show_data_management()
elif page == "Model Prediction":
    show_model_prediction()
