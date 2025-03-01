import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pygame

st.header("This is mine")
# Generate random time series data
if st.button("Test this"):
    def get_model_params():
    return {
        "N": st.sidebar.slider("Number of agents", 50, 500, 100),
        "misinformation_spread_prob": st.sidebar.slider("Misinformation Spread Probability", 0.0, 1.0, 0.3),
        "skeptic_ratio": st.sidebar.slider("Skeptic Ratio", 0.0, 1.0, 0.2),
        "influencer_ratio": st.sidebar.slider("Influencer Ratio", 0.0, 1.0, 0.1),
        "fact_check_prob": st.sidebar.slider("Fact Check Probability", 0.0, 1.0, 0.1),
        "epsilon": st.sidebar.slider("Epsilon (Exploration Rate)", 0.0, 1.0, 0.1),
        "steps": 200,
        "duration": 100  # Duration of simulation in seconds
    }
