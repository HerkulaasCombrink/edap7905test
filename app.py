import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time

# Streamlit UI setup
st.title("Misinformation Dynamic Network Simulation")
st.sidebar.header("Simulation Parameters")

# Customizable parameters
N = st.sidebar.slider("Number of Agents", min_value=50, max_value=500, value=100, step=10)
misinformation_spread_prob = st.sidebar.slider("Misinformation Spread Probability", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
fact_check_prob = st.sidebar.slider("Fact-Checking Probability", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
skeptic_conversion_prob = st.sidebar.slider("Skeptic Conversion Probability", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
epsilon = st.sidebar.slider("Epsilon (E-Greedy Believers)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
steps = st.sidebar.slider("Simulation Steps", min_value=50, max_value=500, value=200, step=10)

# New Parameters for SSI and Stress Propagation
alpha = st.sidebar.slider("Stress Propagation Factor (α)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
beta = st.sidebar.slider("Fact-Check Impact (β)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
gamma = st.sidebar.slider("Misinformation Impact (γ)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
lambda_factor = st.sidebar.slider("Network Effect Factor (λ)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

# Create a Scale-Free Network
G = nx.barabasi_albert_graph(N, 3)
network_pos = nx.spring_layout(G)
SSI = {node: random.uniform(0.1, 0.5) for node in G.nodes()}

# Initialize agent properties
belief_states = ["Believer", "Skeptic", "Neutral", "Influencer"]
node_colors = {node: "gray" for node in G.nodes()}
node_sizes = {node: 80 for node in G.nodes()}
skep_strategies = {}
agent_types = {"Believer": set(), "Skeptic": set(), "Neutral": set(), "Influencer": set()}

# Microblog data
agent_microblogs = {node: [] for node in G.nodes()}

# Assign belief states
for node in G.nodes():
    belief = random.choices(belief_states, weights=[0.4, 0.3, 0.2, 0.1])[0]
    agent_types[belief].add(node)
    if belief == "Skeptic":
        skep_strategies[node] = "UCB"
    elif belief == "Believer":
        skep_strategies[node] = "E-Greedy"
    elif belief == "Influencer":
        node_colors[node] = "green"
        node_sizes[node] = 300
# Function to draw the network
def draw_network(G, node_colors, node_sizes, network_pos):
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw(
        G, pos=network_pos, ax=ax, node_color=[node_colors[node] for node in G.nodes()],
        node_size=[node_sizes[node] for node in G.nodes()], edge_color='gray', alpha=0.5
    )
    plt.title("Network Visualization of Misinformation Dynamics")
    st.pyplot(fig)

# Create a Scale-Free Network
N = 100  # Default number of agents
G = nx.barabasi_albert_graph(N, 3)
network_pos = nx.spring_layout(G)

# Initialize agent properties
belief_states = ["Believer", "Skeptic", "Neutral", "Influencer"]
node_colors = {node: "gray" for node in G.nodes()}
node_sizes = {node: 80 for node in G.nodes()}

# Assign belief states
for node in G.nodes():
    belief = random.choices(belief_states, weights=[0.4, 0.3, 0.2, 0.1])[0]
    if belief == "Believer":
        node_colors[node] = "red"
    elif belief == "Skeptic":
        node_colors[node] = "blue"
    elif belief == "Influencer":
        node_colors[node] = "green"
        node_sizes[node] = 300

# Display the network
draw_network(G, node_colors, node_sizes, network_pos)
# Simulation button
if st.sidebar.button("Start Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    network_plot = st.empty()
    graph_plot = st.empty()

    # Store simulation data
    data_log = []

    for t in range(steps):
        for node in list(G.nodes()):
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            # Microblog-level computations
            microblogs = agent_microblogs[node]
            SA = np.mean([random.uniform(-1, 1) for _ in range(len(microblogs))]) if microblogs else random.uniform(-0.1, 0.1)
            SUB = np.mean([random.uniform(0, 1) for _ in range(len(microblogs))]) if microblogs else random.uniform(0, 0.5)
            ISB = np.mean([random.uniform(0, 1) for _ in range(len(microblogs))]) if microblogs else random.uniform(0, 0.5)
            
            # Compute SSI update
            propagation_effect = (alpha / lambda_factor) * sum(SSI[n] for n in neighbors) / max(1, len(neighbors))
            misinformation_effect = (gamma / lambda_factor) * misinformation_spread_prob
            fact_check_effect = (beta / lambda_factor) * fact_check_prob
            SSI[node] = max(0, min(1, SSI[node] + propagation_effect - fact_check_effect + misinformation_effect))
            
            # Influence belief dynamics
            if node in agent_types["Believer"]:
                if random.random() < misinformation_spread_prob:
                    target = random.choice(neighbors)
                    if target in agent_types["Neutral"]:
                        agent_types["Believer"].add(target)
                        agent_types["Neutral"].remove(target)
                        node_colors[target] = "red"

            elif node in agent_types["Skeptic"]:
                if random.random() < fact_check_prob:
                    target = random.choice(neighbors)
                    if target in agent_types["Believer"]:
                        agent_types["Skeptic"].add(target)
                        agent_types["Believer"].remove(target)
                        node_colors[target] = "blue"

        # Log simulation data
        for node in G.nodes():
            data_log.append([t, node, node_colors[node], SA, SUB, ISB, SSI[node]])

        progress_bar.progress((t + 1) / steps)
        status_text.text(f"Simulation Step {t + 1}/{steps}")

    # Convert log to DataFrame
    df = pd.DataFrame(data_log, columns=["Step", "Agent", "Belief", "SA", "SUB", "ISB", "SSI"])
    
    # Allow CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Simulation Data", csv, "simulation_data.csv", "text/csv")

    st.success("Simulation Complete")
