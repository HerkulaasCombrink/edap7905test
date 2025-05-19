import streamlit as st
import requests

st.title("🌦️ Current Weather App (Public API)")

city = st.text_input("Enter a city", "London")

# Public demo key and base URL
api_key = "439d4b804bc8187953eb36d2a8c26a02"
base_url = "https://api.openweathermap.org/data/2.5/weather"

# API request
params = {
    "q": city,
    "appid": api_key,
    "units": "metric"
}

response = requests.get(base_url, params=params)
data = response.json()

if response.status_code == 200:
    st.subheader(f"Weather in {city}")
    st.write(f"🌡️ Temperature: {data['main']['temp']} °C")
    st.write(f"💧 Humidity: {data['main']['humidity']} %")
    st.write(f"🌬️ Wind Speed: {data['wind']['speed']} m/s")
    st.write(f"🌥️ Conditions: {data['weather'][0]['description'].title()}")
else:
    st.error("City not found or API error.")
