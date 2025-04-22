import streamlit as st
import bnlearn as bn
import matplotlib.pyplot as plt
import ast

# Page configuration
st.set_page_config(page_title="Bayesian Network Builder", layout="centered")

# Title
st.title("🧠 Bayesian Network Builder")
st.markdown("Define your edges and visualise your Bayesian Network using `bnlearn`.")

# Instructions
st.markdown(
    """
### ✍️ How to use
Enter your edges in the format:

```python
[('A', 'B'), ('B', 'C')]
