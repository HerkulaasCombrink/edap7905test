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
    # Build and plot the network
    model = BayesianNetwork(user_edges)
    fig, ax = plt.subplots()
    nx.draw(model, with_labels=True, node_color='lightblue', node_size=2000, font_size=14, arrowsize=20)
    st.pyplot(fig)
else:
    st.warning("⚠️ Please enter a list of 2-element tuples.")
