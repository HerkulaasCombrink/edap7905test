#Network dynamic code

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
skeptic_conversion_prob = st.sidebar.slider("Skeptic Conversion Probability", min_value=0.0, max_value=1.0, value=0.05, step=0.01)  # New parameter
epsilon = st.sidebar.slider("Epsilon (E-Greedy Believers)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
steps = st.sidebar.slider("Simulation Steps", min_value=50, max_value=500, value=200, step=10)
# New Parameters for SSI and Stress Propagation
alpha = st.sidebar.slider("Stress Propagation Factor (α)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
beta = st.sidebar.slider("Fact-Check Impact (β)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
gamma = st.sidebar.slider("Misinformation Impact (γ)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
lambda_factor = st.sidebar.slider("Network Effect Factor (λ)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
# Initialize Social Stress Indicator (SSI) for all nodes
# Simulation button to start execution

# Algorithm selection
believer_algorithm = st.sidebar.selectbox("Believer Strategy", ["E-Greedy", "Thompson Sampling", "UCB", "Random"])
skeptic_algorithm = st.sidebar.selectbox("Skeptic Strategy", ["UCB", "Thompson Sampling", "Random"])


# Create a Scale-Free Network
G = nx.barabasi_albert_graph(N, 3)
network_pos = nx.spring_layout(G)  # Fixed layout for consistent visualization
SSI = {node: random.uniform(0.1, 0.5) for node in G.nodes()}

# Assign belief states to nodes
belief_states = ["Believer", "Skeptic", "Neutral", "Influencer"]
# Initialize node colors before the simulation starts
node_colors = {node: "gray" for node in G.nodes()}  # Default to neutral color
node_sizes = {node: 80 for node in G.nodes()}  # Default to small size
skep_strategies = {}  # Store selected skeptic strategy
agent_types = {"Believer": set(), "Skeptic": set(), "Neutral": set(), "Influencer": set()}
rewards = {"Skeptic": [0], "Believer": [0]}  # Track cumulative rewards over time
# Initialize belief counts for tracking
# Assign belief states to nodes properly before simulation starts
# Assign belief states and strategies
for node in G.nodes():
    belief = random.choices(belief_states, weights=[0.4, 0.3, 0.2, 0.1])[0]
    
    for state in belief_states:  # Ensure all states are initialized
        if state not in agent_types:
            agent_types[state] = set()
    
    agent_types[belief].add(node)

    # Assign strategies properly
    if belief == "Skeptic":
        skep_strategies[node] = skeptic_algorithm
    elif belief == "Believer":
        skep_strategies[node] = believer_algorithm
    elif belief == "Influencer":
        node_colors[node] = "green"
        node_sizes[node] = 300  # Make influencers larger

# Ensure the belief counts are initialized correctly
belief_counts = {
    "Believers": [len(agent_types["Believer"])],
    "Skeptics": [len(agent_types["Skeptic"])],
    "Neutrals": [len(agent_types["Neutral"])],
    "Influencers": [len(agent_types["Influencer"])]
}
# Simulation button to start execution
if st.sidebar.button("Start Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    network_plot = st.empty()
    graph_plot = st.empty()
    
    # Initialize SSI tracking over time
    SSI_over_time = []

    for t in range(steps):
        reward_skeptic = rewards["Skeptic"][-1]
        reward_believer = rewards["Believer"][-1]

        for node in list(G.nodes()):
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            # Update SSI using the propagation model
            propagation_effect = (alpha / lambda_factor) * sum(SSI[n] for n in neighbors) / max(1, len(neighbors))
            misinformation_effect = (gamma / lambda_factor) * misinformation_spread_prob
            fact_check_effect = (beta / lambda_factor) * fact_check_prob

            # Compute SSI for this node
            stress_variation = random.uniform(-0.05, 0.05)  # Adds small variation in stress
            SSI[node] = max(0, min(1, SSI[node] + propagation_effect - fact_check_effect + misinformation_effect + stress_variation))

            if neighbors:
                target = random.choice(neighbors)  # Always set target

                if target in agent_types["Neutral"]:
                    if random.random() < misinformation_spread_prob:
                        agent_types["Believer"].add(target)
                        agent_types["Neutral"].discard(target)
                        node_colors[target] = "red"
                        reward_believer += 1
                        SSI[target] += misinformation_effect * random.uniform(0.8, 1.2)  # Adds variability to stress
                elif random.random() < fact_check_prob:  # Skeptics can prevent conversion
                        agent_types["Skeptic"].add(target)
                        agent_types["Neutral"].discard(target)
                        node_colors[target] = "blue"
                        reward_skeptic += 1
                        SSI[target] -= fact_check_effect  # Reduce stress when fact-checking succeeds

                elif target in agent_types["Skeptic"] and random.random() < (misinformation_spread_prob * 0.5):
                    agent_types["Believer"].add(target)
                    agent_types["Skeptic"].discard(target)
                    node_colors[target] = "red"
                    reward_believer += 1
                    SSI[target] += misinformation_effect
                
                if believer_algorithm == "UCB":
                    if random.random() < misinformation_spread_prob:
                        target = max(neighbors, key=lambda n: len(list(G.neighbors(n))), default=target)

                if random.random() < misinformation_spread_prob and target in agent_types["Neutral"]:
                    agent_types["Believer"].add(target)
                    agent_types["Neutral"].remove(target)
                    node_colors[target] = "red"
                    reward_believer += 1
                    SSI[target] += misinformation_effect  # Increase stress for misinformation spread

                elif target in agent_types["Influencer"]:
                    agent_types["Believer"].add(target)
                    node_colors[target] = "red"
                    reward_believer += 2
                    SSI[target] += misinformation_effect  # Increase stress for misinformation spread

            elif node in agent_types["Skeptic"]:  # Skeptics applying selected strategy
                if skep_strategies.get(node, "UCB") == "UCB":
                    if target in agent_types["Believer"] and random.random() < fact_check_prob * 1.5:  # Higher chance to convert
                        agent_types["Believer"].discard(target)
                        agent_types["Skeptic"].add(target)
                        node_colors[target] = "blue"
                        reward_skeptic += 1
                        SSI[target] = max(0, SSI[target] - (fact_check_effect * random.uniform(0.8, 1.2)))  # Reduce stress

                elif skep_strategies.get(node, "UCB") == "Thompson Sampling":
                    if random.betavariate(2, 5) > 0.5 and target in agent_types["Believer"]:
                        agent_types["Believer"].remove(target)
                        agent_types["Skeptic"].add(target)
                        skep_strategies[target] = "UCB"
                        node_colors[target] = "blue"
                        reward_skeptic += 1
                        SSI[target] -= fact_check_effect  # Reduce stress for successful fact-checking

                elif skep_strategies.get(node, "UCB") == "Random":
                    if random.random() < 0.5 and target in agent_types["Believer"]:
                        agent_types["Believer"].remove(target)
                        agent_types["Skeptic"].add(target)
                        node_colors[target] = "blue"
                        reward_skeptic += 1
                        SSI[target] -= fact_check_effect  # Reduce stress for successful fact-checking

                # Skeptic conversion back to Neutral
                if node in agent_types["Skeptic"] and random.random() < skeptic_conversion_prob:
                    agent_types["Skeptic"].discard(node)  # Use discard to avoid KeyErrors
                    agent_types["Neutral"].add(node)
                    node_colors[node] = "gray"     

        # Track rewards and belief updates
        rewards["Believer"].append(reward_believer)
        rewards["Skeptic"].append(reward_skeptic)

        for key in ["Believers", "Skeptics", "Neutrals", "Influencers"]:
            if key not in belief_counts:
                belief_counts[key] = []  # Initialize key if missing

        belief_counts["Believers"].append(len(agent_types["Believer"]))
        belief_counts["Skeptics"].append(len(agent_types["Skeptic"]))
        belief_counts["Neutrals"].append(len(agent_types["Neutral"]))
        belief_counts["Influencers"].append(len(agent_types["Influencer"]))
        SSI_over_time.append(np.mean(list(SSI.values())))

        # Update progress bar
        progress_bar.progress((t + 1) / steps)
        status_text.text(f"Simulation Step {t + 1}/{steps}")

        if t % 10 == 0:  # Update visualization every 10 steps
            fig, ax = plt.subplots(figsize=(12, 10))
            
            node_colors = {
                node: "red" if node in agent_types["Believer"] else
                    "blue" if node in agent_types["Skeptic"] else
                    "green" if node in agent_types["Influencer"] else
                    "gray" for node in G.nodes()
}

            fig, ax = plt.subplots(figsize=(12, 10))
            nx.draw(G, pos=network_pos, 
                node_color=[node_colors[n] for n in G.nodes()], 
                node_size=[100 + 300 * SSI[n] for n in G.nodes()], 
                edge_color="gray", with_labels=False)
            network_plot.pyplot(fig)

# Time series graphs update every 10 steps
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].plot(range(len(belief_counts["Believers"])), belief_counts["Believers"], label="Believers", color="red")
            axs[0].plot(range(len(belief_counts["Skeptics"])), belief_counts["Skeptics"], label="Skeptics", color="blue")
            axs[0].set_title("Believers vs. Skeptics Over Time")
            axs[0].legend()

            axs[1].plot(range(len(belief_counts["Neutrals"])), belief_counts["Neutrals"], label="Neutrals", color="gray")
            axs[1].set_title("Neutral Count Over Time")
            axs[1].legend()

            axs[2].plot(range(len(SSI_over_time)), SSI_over_time, label="SSI Over Time", color="black")
            axs[2].set_title("Social Stress Indicator (SSI) Over Time")
            axs[2].legend()

            graph_plot.pyplot(fig)

    st.success("Simulation Complete")

