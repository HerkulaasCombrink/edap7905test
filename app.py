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
user_input = st.text_area("✍️ Type your edge list here:")
if user_input: try: 
    # Validate and construct network
    if isinstance(parsed, list) and all(isinstance(e, tuple) and len(e) == 2 for e in parsed):
        st.success("✅ Edges parsed successfully!")

        # Create the Bayesian Network
        model = BayesianNetwork(parsed)

        # Draw the network
        fig, ax = plt.subplots()
        nx.draw(
            model,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=14,
            font_weight="bold",
            arrows=True,
            arrowsize=20
        )
        st.pyplot(fig)
    else:
        st.warning("⚠️ Input must be a list of 2-element tuples like [('A', 'B')].")

except Exception as e:
    st.error(f"❌ Could not parse your input. Please check the format.\n\n**Error:** {e}")
