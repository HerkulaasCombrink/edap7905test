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
if isinstance(user_edges, list) and all(isinstance(e, tuple) and len(e) == 2 for e in user_edges):
    st.success("✅ Edges parsed successfully!")

    # Build Bayesian DAG
    model = bn.make_DAG(user_edges)

    # Plot the graph
    fig = bn.plot(model, verbose=0)
    st.pyplot(fig)

else:
    st.warning("⚠️ Please enter a list of tuples with two elements each, like [('A', 'B')].")
