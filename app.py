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
# Function to draw the network
def draw_network(G, node_colors, node_sizes, network_pos, plot_area):
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw(
        G, pos=network_pos, ax=ax, node_color=[node_colors[node] for node in G.nodes()],
        node_size=[node_sizes[node] for node in G.nodes()], edge_color='gray', alpha=0.5
    )
    plt.title("Network Visualization of Misinformation Dynamics")
    plot_area.pyplot(fig)

# Function to plot time series graphs
def plot_time_series(data_log, plot_area):
    df = pd.DataFrame(data_log, columns=["Step", "Believers", "Skeptics", "Neutrals", "Influencers"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["Believers", "Skeptics", "Neutrals", "Influencers"]:
        ax.plot(df["Step"], df[col], label=col, linewidth=1)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Agent Count")
    ax.set_title("Evolution of Agent Belief States Over Time")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))  # Fix legend position
    plot_area.pyplot(fig)

# Simulation button
if st.sidebar.button("Start Simulation"):
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
    agent_microblogs = {node: [] for node in G.nodes()}

    # **Move UCB Initialization Inside the Button Block**
    ucb_counts = {node: 1 for node in G.nodes()}  # Count of times each node has been influenced
    ucb_values = {node: 0 for node in G.nodes()}  # UCB estimated values

    # Draw initial network visualization
    network_plot = st.empty()
    time_series_plot = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    draw_network(G, node_colors, node_sizes, network_pos, network_plot)

    # Data log for time series plotting
    data_log = []

    for t in range(steps):
        for node in list(G.nodes()):  # Ensure we iterate over a copy to modify the structure safely
            if node in agent_types["Influencer"] or node in agent_types["Believer"] or node in agent_types["Skeptic"]:
                neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

        # Apply UCB to select a neighbor to influence
            ucb_scores = {n: ucb_values[n] + np.sqrt(2 * np.log(sum(ucb_counts.values())) / ucb_counts[n]) for n in neighbors}
            target = max(ucb_scores, key=ucb_scores.get)  # Select neighbor with highest UCB score
        
        # **Updated: Ensure Proper Belief Transition**
            if target in agent_types["Neutral"]:
                if node in agent_types["Believer"]:
                    agent_types["Believer"].add(target)
                    agent_types["Neutral"].remove(target)
                    node_colors[target] = "red"
                elif node in agent_types["Skeptic"]:
                    agent_types["Skeptic"].add(target)
                    agent_types["Neutral"].remove(target)
                    node_colors[target] = "blue"
        
                elif target in agent_types["Believer"] and node in agent_types["Skeptic"]:
                    agent_types["Skeptic"].add(target)
                    agent_types["Believer"].remove(target)
                    node_colors[target] = "blue"

                elif target in agent_types["Skeptic"] and node in agent_types["Believer"]:
                    agent_types["Believer"].add(target)
                    agent_types["Skeptic"].remove(target)
                    node_colors[target] = "red"

        # **Ensure influencers can impact multiple nodes**
        if node in agent_types["Influencer"]:
            for neighbor in neighbors:
                if neighbor in agent_types["Neutral"]:
                    agent_types["Believer"].add(neighbor)
                    agent_types["Neutral"].remove(neighbor)
                    node_colors[neighbor] = "red"

        # **Update UCB values**
        reward = 1 if target in agent_types["Believer"] else 0  # Reward when a neutral becomes a believer
        ucb_values[target] = (ucb_values[target] * ucb_counts[target] + reward) / (ucb_counts[target] + 1)
        ucb_counts[target] += 1

                # Apply UCB to select a neighbor to influence
        ucb_scores = {n: ucb_values[n] + np.sqrt(2 * np.log(sum(ucb_counts.values())) / ucb_counts[n]) for n in neighbors}
        target = max(ucb_scores, key=ucb_scores.get)  # Select neighbor with highest UCB score
                
                # Determine belief transition
                if node in agent_types["Believer"] and target in agent_types["Neutral"]:
                    agent_types["Believer"].add(target)
                    agent_types["Neutral"].remove(target)
                    node_colors[target] = "red"
                elif node in agent_types["Skeptic"] and target in agent_types["Believer"]:
                    agent_types["Skeptic"].add(target)
                    agent_types["Believer"].remove(target)
                    node_colors[target] = "blue"

                # Update UCB values
                reward = 1 if target in agent_types["Believer"] else 0  # Reward when a neutral becomes a believer
                ucb_values[target] = (ucb_values[target] * ucb_counts[target] + reward) / (ucb_counts[target] + 1)
                ucb_counts[target] += 1

        # Log data for time series graph
        data_log.append([
            t, len(agent_types["Believer"]), len(agent_types["Skeptic"]),
            len(agent_types["Neutral"]), len(agent_types["Influencer"])
        ])

        # Update visualization every update_interval steps
        if t % 10 == 0:
            draw_network(G, node_colors, node_sizes, network_pos, network_plot)
            plot_time_series(data_log, time_series_plot)
            time.sleep(0.5)

        progress_bar.progress((t + 1) / steps)
        status_text.text(f"Simulation Step {t + 1}/{steps}")

    st.success("Simulation Complete")
    
    st.success("Simulation Complete")
