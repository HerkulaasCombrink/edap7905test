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
    
    # Streamlit placeholders
    network_plot = st.empty()
    time_series_plot = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data log for time series plotting
    data_log = []
    
    for t in range(steps):
        for node in G.nodes():
            if random.random() < 0.05:
                new_belief = random.choice(belief_states)
                for b in belief_states:
                    agent_types[b].discard(node)
                agent_types[new_belief].add(node)
                if new_belief == "Believer":
                    node_colors[node] = "red"
                elif new_belief == "Skeptic":
                    node_colors[node] = "blue"
                elif new_belief == "Influencer":
                    node_colors[node] = "green"
                    node_sizes[node] = 300
                else:
                    node_colors[node] = "gray"
                    node_sizes[node] = 80
        
        # Log data for time series graph
        data_log.append([
            t, len(agent_types["Believer"]), len(agent_types["Skeptic"]),
            len(agent_types["Neutral"]), len(agent_types["Influencer"])
        ])
        
        # Update visualization every update_interval steps
        if t % 10 == 0:
            network_plot.empty()
            time_series_plot.empty()
            time.sleep(0.5)
        
        progress_bar.progress((t + 1) / steps)
        status_text.text(f"Simulation Step {t + 1}/{steps}")
    
    st.success("Simulation Complete")
