import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
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
if user_input:
    try:
        parsed = ast.literal_eval(user_input)
        if isinstance(parsed, list) and all(isinstance(e, tuple) and len(e) == 2 for e in parsed):
            st.success("✅ Edges parsed successfully!")

            # Build the Bayesian network (optional if you just want to draw)
            model = DiscreteBayesianNetwork(parsed)

            # Build a NetworkX DiGraph for visualization
            G = nx.DiGraph()
            G.add_edges_from(parsed)

            # Draw using a spring layout
            pos = nx.spring_layout(G)
            fig, ax = plt.subplots()
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)
            nx.draw_networkx_edges(
                G, pos,
                arrowstyle='-|>',
                arrowsize=20,
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)

        else:
            st.warning("⚠️ Input must be a list of 2-element tuples like [('A', 'B')].")

    except Exception as e:
        st.error(f"❌ Could not parse your input. Please check the format.\n\n**Error:** {e}")
