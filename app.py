import streamlit as st
#import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#import random
#import time
#import pygame

st.header("This is mine")
# Generate random time series data
if st.button("Test this"):
  time_series = np.random.randn(100)
