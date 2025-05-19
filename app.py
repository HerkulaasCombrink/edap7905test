import pandas as pd
import streamlit as st

# Public CSV file from GitHub
url = "https://github.com/HerkulaasCombrink/edap7905test/blob/main/data.csv"
df = pd.read_csv(url)
st.dataframe(df)
