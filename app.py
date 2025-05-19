import streamlit as st
import requests

st.title("🐶 Random Dog Image Viewer")

if st.button("Get a Dog"):
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    
    if response.status_code == 200:
        data = response.json()
        st.image(data['message'], caption="Here's a random dog!", use_column_width=True)
    else:
        st.error("Could not fetch dog image.")
