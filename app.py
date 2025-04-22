import streamlit as st
from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import ast

# Configure Streamlit page
st.set_page_config(page_title="Bayesian Network Builder", layout="centered")

# App title and instructions
st.title("🧠 Bayesian Network Builder")

st.markdown(
    """
Build and visualise your own Bayesian Network using `pgmpy`.

### ✍️ Instructions:
- Enter your edges below in Python list format.
- Each tuple represents a directed edge from one node to another.

Example:
```python
[('A', 'B'), ('B', 'C')]
""")
# Validate structure
if isinstance(user_edges, list) and all(isinstance(e, tuple) and len(e) == 2 for e in user_edges):
    st.success("✅ Edges parsed successfully!")

    # Build Bayesian Network
    model = BayesianNetwork(user_edges)

    # Plot with networkx
    fig, ax = plt.subplots()
    nx.draw(
        model,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=14,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
    )
    st.pyplot(fig)
else:
    st.warning("⚠️ Input must be a list of 2-element tuples like [('A', 'B')].")
