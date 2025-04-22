import streamlit as st
from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import ast

st.set_page_config(page_title="Bayesian Network Builder", layout="centered")

st.title("🧠 Bayesian Network Builder (pgmpy version)")
st.markdown("""
Enter your edges in the format:

```python
[('A', 'B'), ('B', 'C')]
