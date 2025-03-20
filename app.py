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
    skeptic_duration = {node: 0 for node in G.nodes()}  
    misfluencer_duration = {node: 0 for node in G.nodes()}  

# Base probabilities
    base_conversion_prob = 0.1
    misfluencer_easy_conversion = 0.4  # Misfluencers convert more easily
    skeptic_resistance_factor = 0.5  # Skeptics take longer to change
    agent_microblogs = {node: [] for node in G.nodes()}

    # **Move Initial Agent Assignment Here**
    all_nodes = list(G.nodes())
    random.shuffle(all_nodes)  # Shuffle to ensure randomness
    num_believers = max(1, int(0.2 * N))   # 20% believers
    num_skeptics = max(1, int(0.2 * N))    # 20% skeptics
    num_influencers = max(1, int(0.05 * N)) # 5% influencers
    num_neutrals = N - (num_believers + num_skeptics + num_influencers)  # Remaining are neutrals
    
    # Assign agent types
    believers = set(all_nodes[:num_believers])
    skeptics = set(all_nodes[num_believers:num_believers + num_skeptics])
    influencers = set(all_nodes[num_believers + num_skeptics:num_believers + num_skeptics + num_influencers])
    neutrals = set(all_nodes[num_believers + num_skeptics + num_influencers:])

    # Apply assignments
    agent_types["Believer"] = believers
    agent_types["Skeptic"] = skeptics
    agent_types["Influencer"] = influencers
    agent_types["Neutral"] = neutrals

    # Assign colors
    for node in believers:
        node_colors[node] = "red"
    for node in skeptics:
        node_colors[node] = "blue"
    for node in influencers:
        node_colors[node] = "green"
        node_sizes[node] = 300
    for node in neutrals:
        node_colors[node] = "gray"

    # Streamlit placeholders
    network_plot = st.empty()
    time_series_plot = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Draw initial network visualization
    draw_network(G, node_colors, node_sizes, network_pos, network_plot)

    # Data log for time series plotting
    data_log = []
    # ✅ Add UCB initialization here

   
# Initialize UCB parameters for all nodes before simulation starts
    ucb_counts = {node: 1 for node in G.nodes()}  # Track how often each node has been influenced
    ucb_values = {node: 0.5 for node in G.nodes()}  # Start with a neutral belief score
    exploration_factor = 2.5  # Encourage exploration
    penalty = -0.2  # Discourage inaction
    for t in range(steps):
        for node in list(G.nodes()):
            neighbors = list(G.neighbors(node))  
            if not neighbors:
                continue  

    # Dynamic conversion probability based on network conditions
            believers_count = len(agent_types["Believer"])
            skeptics_count = len(agent_types["Skeptic"])
            stress_factor = abs(believers_count - skeptics_count) / max(1, N)  

    # Choose a target neighbor for influence
        neutral_neighbors = [n for n in neighbors if n in agent_types["Neutral"]]
        if neutral_neighbors:
            target = random.choice(neutral_neighbors)
                # Ensure conversion success rate increases over time
            conversion_boost = min(1.0, 0.2 + (t / steps) * 0.5)  # Becomes easier over time
            if node in agent_types["Believer"]:
                if random.random() < (misinformation_spread_prob + conversion_boost):
                    agent_types["Believer"].add(target)
                    agent_types["Neutral"].remove(target)
                    node_colors[target] = "red"
                elif node in agent_types["Skeptic"]:
                    if random.random() < (fact_check_prob + conversion_boost):
                        agent_types["Skeptic"].add(target)
                        agent_types["Neutral"].remove(target)
                        node_colors[target] = "blue"
            else:
                target = random.choice(neighbors)  # Engage with other non-neutrals  
                reward = 1 if target in agent_types["Believer"] else -0.2
                ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
                ucb_counts[target] += 1
    # Convert misfluencers easily
            if node in agent_types["Believer"] and target in agent_types["Influencer"]:
                if random.random() < misfluencer_easy_conversion * (1 + stress_factor):
                    agent_types["Believer"].add(target)
                    agent_types["Influencer"].remove(target)
                    node_colors[target] = "red"

            elif node in agent_types["Skeptic"] and target in agent_types["Influencer"]:
                if random.random() < misfluencer_easy_conversion * (1 + stress_factor):
                    agent_types["Skeptic"].add(target)
                    agent_types["Influencer"].remove(target)
                    node_colors[target] = "blue"

    # Skeptics resist more but can change
                if node in agent_types["Believer"] and target in agent_types["Skeptic"]:
                    skeptic_duration[target] += 1  
                    if skeptic_duration[target] > 3 and random.random() < base_conversion_prob * (1 - skeptic_resistance_factor):
                        agent_types["Believer"].add(target)
                        agent_types["Skeptic"].remove(target)
                        node_colors[target] = "red"
                        skeptic_duration[target] = 0  # Reset

                    elif node in agent_types["Skeptic"] and target in agent_types["Believer"]:
                        if random.random() < base_conversion_prob * (1 + skeptic_resistance_factor):
                            agent_types["Skeptic"].add(target)
                            agent_types["Believer"].remove(target)
                            node_colors[target] = "blue"

    # Misfluencers change fast
                if node in agent_types["Influencer"]:
                    misfluencer_duration[node] += 1
                    if misfluencer_duration[node] > 5:
                        if random.random() < 0.5:
                            agent_types["Believer"].add(node)
                            agent_types["Influencer"].remove(node)
                            node_colors[node] = "red"
                        else:
                            agent_types["Skeptic"].add(node)
                            agent_types["Influencer"].remove(node)
                            node_colors[node] = "blue"
                        misfluencer_duration[node] = 0

            # Allow multiple influence attempts per step
                influence_attempts = min(5, max(2, len(agent_types["Neutral"]) // 10))  # More attempts if more neutrals exist
                for _ in range(influence_attempts):
                    ucb_scores = {n: ucb_values[n] + exploration_factor * np.sqrt(np.log(sum(ucb_counts.values()) + 1) / ucb_counts[n]) + penalty for n in neighbors}
                    target = max(ucb_scores, key=ucb_scores.get)
                    for n in neighbors:
                        if ucb_counts[n] == 0:  # Prevent division by zero
                            ucb_counts[n] = 1
                        ucb_scores[n] = ucb_values[n] + exploration_factor * np.sqrt(np.log(sum(ucb_counts.values()) + 1) / ucb_counts[n]) + penalty

                    target = max(ucb_scores, key=ucb_scores.get)

                # **Influence Spreading Logic**
                if target in agent_types["Neutral"]:
                    conversion_prob = misinformation_spread_prob if node in agent_types["Believer"] else fact_check_prob
                    conversion_boost = 0.2 if node in agent_types["Influencer"] else 0  # Influencers boost conversion
                    if random.random() < (conversion_prob + conversion_boost):
                        if node in agent_types["Believer"]:
                            agent_types["Believer"].add(target)
                            agent_types["Neutral"].remove(target)
                            node_colors[target] = "red"
                        elif node in agent_types["Skeptic"]:
                            agent_types["Skeptic"].add(target)
                            agent_types["Neutral"].remove(target)
                            node_colors[target] = "blue"

                # Influence multiple nodes if influencer
                if node in agent_types["Influencer"]:
                    for neighbor in neighbors:
                        if neighbor in agent_types["Neutral"]:
                            if random.random() < 0.9:  # 90% chance of conversion
                                agent_types["Believer"].add(neighbor)
                                agent_types["Neutral"].remove(neighbor)
                                node_colors[neighbor] = "red"

# Ensure simulation does not stop early
        if len(agent_types["Neutral"]) > 0:
            continue  # Prevent early stopping

                # UCB update
            reward = 1 if target in agent_types["Believer"] else -0.2  # Give negative reward if no influence
            ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
            ucb_counts[target] += 1  # Increase count after update

    if node in agent_types["Believer"] or node in agent_types["Skeptic"] or node in agent_types["Influencer"]:
            # UCB scoring for influence choice
                ucb_scores = {}
                exploration_factor = 2.5  # Encourages exploration
                penalty = -0.2
                for n in neighbors:
                    if ucb_counts[n] == 0:  # Prevent division by zero
                        ucb_counts[n] = 1
                    ucb_scores[n] = ucb_values[n] + exploration_factor * np.sqrt(np.log(sum(ucb_counts.values()) + 1) / ucb_counts[n]) + penalty
    # Introduce a higher exploration factor and a small penalty for no action
                    exploration_factor = 2.5  # Increase exploration tendency
                    penalty = -0.2  # Discourage repeated inaction
                    ucb_scores[n] = ucb_values[n] + exploration_factor * np.sqrt(np.log(sum(ucb_counts.values()) + 1) / ucb_counts[n]) + penalty

# Select the neighbor with the highest UCB score
                target = max(ucb_scores, key=ucb_scores.get)

            # **Influence Logic**
                if target in agent_types["Neutral"]:
                    if node in agent_types["Believer"]:
                        if random.random() < misinformation_spread_prob:
                            agent_types["Believer"].add(target)
                            agent_types["Neutral"].remove(target)
                            node_colors[target] = "red"
                    elif node in agent_types["Skeptic"]:
                        if random.random() < fact_check_prob:
                            agent_types["Skeptic"].add(target)
                            agent_types["Neutral"].remove(target)
                            node_colors[target] = "blue"
                elif target in agent_types["Skeptic"] and node in agent_types["Believer"]:
                    if skeptic_duration[target] > 3 and random.random() < skeptic_conversion_prob * (1 - skeptic_resistance_factor):
                        agent_types["Believer"].add(target)
                        agent_types["Skeptic"].remove(target)
                        node_colors[target] = "red"
                        skeptic_duration[target] = 0  # Reset conversion cooldown

                # Skeptics converting believers with momentum penalty
                elif node in agent_types["Skeptic"] and target in agent_types["Believer"]:
                    if random.random() < skeptic_conversion_prob * (1 + skeptic_resistance_factor):
                        agent_types["Skeptic"].add(target)
                        agent_types["Believer"].remove(target)
                        node_colors[target] = "blue"
                        skeptic_duration[target] += 2  # Increase resistance time
            # **Influencer Impacts Multiple Nodes**
                if node in agent_types["Influencer"]:
                    for neighbor in neighbors:
                        if neighbor in agent_types["Neutral"]:
                            agent_types["Believer"].add(neighbor)
                            agent_types["Neutral"].remove(neighbor)
                            node_colors[neighbor] = "red"

            # **UCB Update**
                reward = 1 if target in agent_types["Believer"] else 0  # Reward when a neutral becomes a believer
                ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
                ucb_counts[target] += 1  # Increase count after update
 
            # Apply UCB to select a neighbor to influence
                ucb_scores = {}
                for n in neighbors:
                    if ucb_counts[n] == 0:  # Prevent division by zero
                        ucb_counts[n] = 1
                    ucb_scores[n] = ucb_values[n] + np.sqrt(2 * np.log(sum(ucb_counts.values())) / ucb_counts[n])

                    target = max(ucb_scores, key=ucb_scores.get)  # Select best neighbor to influence

            # **Influence Spreading Logic**
                if target in agent_types["Neutral"]:
                    if node in agent_types["Believer"]:
                        agent_types["Believer"].add(target)
                        agent_types["Neutral"].remove(target)
                        node_colors[target] = "red"
                    elif node in agent_types["Skeptic"]:
                        agent_types["Skeptic"].add(target)
                        agent_types["Neutral"].remove(target)
                        node_colors[target] = "blue"

            # **Influencer-Specific Influence Spread**
                if node in agent_types["Influencer"]:
                    for neighbor in neighbors:
                     if neighbor in agent_types["Neutral"]:
                        agent_types["Believer"].add(neighbor)
                        agent_types["Neutral"].remove(neighbor)
                        node_colors[neighbor] = "red"

            # **UCB Value Update**
                        reward = 1 if target in agent_types["Believer"] else 0  # Reward when a neutral becomes a believer
                        ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
                        ucb_counts[target] += 1  # Increase count after update
    
        # Apply UCB to select a neighbor to influence
                        ucb_scores = {}
                for n in neighbors:
                    if ucb_counts[n] == 0:  # Prevent division by zero
                        ucb_counts[n] = 1
                    ucb_scores[n] = ucb_values[n] + np.sqrt(2 * np.log(sum(ucb_counts.values())) / ucb_counts[n])

# Select the neighbor with the highest UCB score
                target = max(ucb_scores, key=ucb_scores.get)
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
                            if random.random() < 0.8:  # Influencers have an 80% chance to convert a neutral
                                agent_types["Believer"].add(neighbor)
                                agent_types["Neutral"].remove(neighbor)
                                node_colors[neighbor] = "red"

        # **Update UCB values**
                reward = 1 if target in agent_types["Believer"] else 0  # Reward when a neutral becomes a believer
                ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
                ucb_counts[target] += 1  # Increase count after updatee

                # Apply UCB to select a neighbor to influence
                ucb_scores = {}
                for n in neighbors:
                    if ucb_counts[n] == 0:  # Prevent division by zero
                        ucb_counts[n] = 1
                    ucb_scores[n] = ucb_values[n] + np.sqrt(2 * np.log(sum(ucb_counts.values())) / ucb_counts[n])

# Select the neighbor with the highest UCB score
        
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
                if target in agent_types["Believer"]:
                    reward = 1.5 if node in agent_types["Influencer"] else 1  # Influencers get higher rewards
                elif target in agent_types["Skeptic"]:
                    reward = 1.5 if node in agent_types["Influencer"] else 1
                elif target in agent_types["Neutral"]:  # Large reward for converting a neutral
                    reward = 2 if node in agent_types["Influencer"] else 1.5

                ucb_values[target] = ((ucb_values[target] * ucb_counts[target]) + reward) / (ucb_counts[target] + 1)
                ucb_counts[target] += 1  # Increase count after update

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

