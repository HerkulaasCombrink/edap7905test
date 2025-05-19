import pandas as pd
import streamlit as st

# Public CSV file from GitHub
url = "https://raw.githubusercontent.com/username/repo/main/data.csv"
df = pd.read_csv(url)
st.dataframe(df)
