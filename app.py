import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# App title
st.title("This is a test app")

# Description
st.write("This is a simulation for a random time series graph. Click the first button to generate the graph, then the second button will be enabled to download the data.")

# Session state to manage button activation
if 'data' not in st.session_state:
    st.session_state.data = None
if 'button2_enabled' not in st.session_state:
    st.session_state.button2_enabled = False

# Button 1 - Generate Random Time Series Graph
generate_button = st.button("Generate Time Series Graph")
if generate_button:
    # Generate random time series data
    date_range = pd.date_range(start='2024-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()  # Random walk simulation
    df = pd.DataFrame({'Date': date_range, 'Value': values})
    st.session_state.data = df
    st.session_state.button2_enabled = True  # Enable button 2

    # Plot the time series
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Value'], linestyle='-', marker='', color='b')
    ax.set_title("Random Time Series Graph")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# Button 2 - Download CSV (only enabled after Button 1 is clicked)
if st.session_state.button2_enabled:
    csv_button = st.download_button(
        label="Download Time Series Data",
        data=st.session_state.data.to_csv(index=False).encode('utf-8'),
        file_name="random_time_series.csv",
        mime="text/csv"
    )

# Add a cool image
st.image("https://source.unsplash.com/800x400/?data,technology", caption="Random Data Visualization")
