import streamlit as st
import xgboost

# Display the XGBoost version in Streamlit
st.sidebar.header("Environment Info")
st.sidebar.write(f"**XGBoost Version:** {xgboost.__version__}")
