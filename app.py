import pandas as pd
import streamlit as st

# ✅ Correct Raw CSV URL from GitHub
url = "https://raw.githubusercontent.com/HerkulaasCombrink/edap7905test/main/data.csv"

# Load CSV into DataFrame
df = pd.read_csv(url)

# Display the data
st.title("📄 GitHub CSV Data Preview")
st.dataframe(df)
