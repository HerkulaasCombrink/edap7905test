import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go
from pytrends.request import TrendReq

# --- Page Configuration ---
st.set_page_config(page_title="Real-Time Trading Dashboard", layout="wide")

# --- Sidebar Controls ---
st.sidebar.title("Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT)", "AAPL")

# Time Range
st.sidebar.subheader("Time Period")
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"], index=6)

# Google Trends Keyword
st.sidebar.subheader("Google Trends")
google_keyword = st.sidebar.text_input("Enter keyword for Google Trends", ticker_symbol)

# --- Load Stock Data ---
st.title("Real-Time Trading Dashboard")
st.write(f"Showing data for **{ticker_symbol}** with period `{period}` and interval `{interval}`")

@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

data = load_stock_data(ticker_symbol)

# --- Technical Indicators ---
def add_indicators(df):
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data = add_indicators(data)

# --- Candlestick Chart ---
st.subheader("Candlestick Chart with Indicators")
candlestick = go.Figure()
candlestick.add_trace(go.Candlestick(x=data.index,
                                     open=data["Open"],
                                     high=data["High"],
                                     low=data["Low"],
                                     close=data["Close"], name="Candlestick"))
candlestick.add_trace(go.Scatter(x=data.index, y=data["SMA20"], line=dict(color='blue', width=1), name="SMA20"))
candlestick.add_trace(go.Scatter(x=data.index, y=data["SMA50"], line=dict(color='orange', width=1), name="SMA50"))
candlestick.update_layout(xaxis_rangeslider_visible=False, height=600)
st.plotly_chart(candlestick, use_container_width=True)

# --- RSI Indicator ---
st.subheader("Relative Strength Index (RSI)")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], line=dict(color='purple', width=1), name="RSI"))
rsi_fig.update_layout(height=300, yaxis_title="RSI")
st.plotly_chart(rsi_fig, use_container_width=True)

# --- Google Trends ---
st.subheader(f"Google Trends for '{google_keyword}'")
@st.cache_data(ttl=3600)
def load_google_trends(keyword):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe='now 7-d')
    df = pytrends.interest_over_time()
    return df.reset_index()

try:
    trends_df = load_google_trends(google_keyword)
    st.line_chart(trends_df.set_index("date")[google_keyword])
except Exception as e:
    st.warning(f"Google Trends data could not be loaded: {e}")

# --- Raw Data Table ---
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Stock Data")
    st.dataframe(data.tail(100))
