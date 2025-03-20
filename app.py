import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# Customizable Parameters
def get_model_params():
    return {
        "N": 100,  # Number of agents
        "smart_agents": 20,  # Number of RL agents
        "network_type": "dynamic",  # Network structure
        "misinformation_spread_prob": 0.3,  # Probability of misinformation spreading
        "skeptic_ratio": 0.2,  # Initial ratio of skeptical agents
        "influencer_ratio": 0.1,  # Ratio of influencers
        "fact_check_prob": 0.1,  # Probability of fact-checking intervention
        "steps": 200,  # Total number of steps
        "rewiring_prob": 0.1,  # Probability of rewiring connections at each step
        "epsilon": 0.1  # Exploration parameter for E-Greedy skeptics
    }

# Create a Dynamic Network
params = get_model_params()
N = params["N"]
rewiring_prob = params["rewiring_prob"]
G = nx.erdos_renyi_graph(N, 0.05)  # Initial random network

# Assign belief states to nodes
belief_states = ["Believer", "Skeptic", "Neutral", "Influencer"]
node_colors = {}
node_sizes = {}
skep_types = ["UCB", "E-Greedy"]
skep_strategies = {}
agent_types = {"UCBAgent": {}, "EpsilonGreedyAgent": {}, "ThompsonSamplingAgent": {}, "RandomAgent": {}}

for node in G.nodes():
    belief = random.choices(belief_states, weights=[0.4, 0.3, 0.2, 0.1])[0]
    agent_type = random.choice(list(agent_types.keys()))
    if belief == "Skeptic":
        skep_strategies[node] = random.choice(skep_types)  # Assign UCB or E-Greedy
    agent_types[agent_type][node] = belief
    node_colors[node] = {"Believer": "red", "Skeptic": "blue", "Neutral": "gray", "Influencer": "green"}[belief]
    node_sizes[node] = {"Believer": 100, "Skeptic": 100, "Neutral": 80, "Influencer": 300}[belief]

# Track success rates for UCB skeptics
ucb_success_rates = {node: 0 for node in G.nodes() if node in skep_strategies and skep_strategies[node] == "UCB"}

# Initialize tracking metrics
belief_counts = {agent: {"Believers": [], "Skeptics": [], "Neutrals": [], "Influencers": []} for agent in agent_types.keys()}

# Function to dynamically rewire edges at each step
def rewire_network(G, prob):
    edges = list(G.edges())
    for edge in edges:
        if random.random() < prob:
            G.remove_edge(*edge)
            new_node = random.choice(list(G.nodes()))
            if new_node != edge[0]:
                G.add_edge(edge[0], new_node)

# RL Agents Implementation
def apply_rl_agents():
    for t in range(params["steps"]):
        rewire_network(G, rewiring_prob)  # Rewire network dynamically

        for agent_type, agent_nodes in agent_types.items():
            for node in agent_nodes.keys():
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue
                target = random.choice(neighbors)

                if node_colors[node] == "red" and node_colors[target] == "gray":
                    if random.random() < params["misinformation_spread_prob"]:
                        node_colors[target] = "red"
                elif node_colors[node] == "blue" and node_colors[target] == "red":
                    if node in skep_strategies:
                        if skep_strategies[node] == "UCB":
                            success_prob = ucb_success_rates.get(node, 0.5)  # Use stored success rate
                            if random.random() < success_prob:
                                node_colors[target] = "gray"
                                ucb_success_rates[node] = (ucb_success_rates[node] + 1) / 2  # Update success probability
                        elif skep_strategies[node] == "E-Greedy":
                            if random.random() < params["epsilon"]:
                                target = random.choice(neighbors)  # Explore new target
                            if random.random() < params["fact_check_prob"]:
                                node_colors[target] = "gray"

            belief_counts[agent_type]["Believers"].append(sum(1 for n in agent_nodes if node_colors[n] == "red"))
            belief_counts[agent_type]["Skeptics"].append(sum(1 for n in agent_nodes if node_colors[n] == "blue"))
            belief_counts[agent_type]["Neutrals"].append(sum(1 for n in agent_nodes if node_colors[n] == "gray"))
            belief_counts[agent_type]["Influencers"].append(sum(1 for n in agent_nodes if node_colors[n] == "green"))

apply_rl_agents()

# Visualization
fig, axs = plt.subplots(4, 4, figsize=(20, 20))

# Plot graphs for each agent type and belief type with 95% confidence intervals
for i, agent_type in enumerate(agent_types.keys()):
    for j, belief_type in enumerate(["Believers", "Skeptics", "Neutrals", "Influencers"]):
        ax = axs[i, j]
        data = np.array(belief_counts[agent_type][belief_type])
        if len(data) > 1:
            mean = np.mean(data, axis=0)
            std_error = stats.sem(data, axis=0)
            confidence_interval = 1.96 * std_error
            ax.plot(range(len(data)), data, label=f"{agent_type} - {belief_type}")
            ax.fill_between(range(len(data)), data - confidence_interval, data + confidence_interval, color='gray', alpha=0.2)
        else:
            ax.plot(range(len(data)), data, label=f"{agent_type} - {belief_type}")
        ax.set_title(f"{agent_type} - {belief_type} Over Time")
        ax.set_xlabel("Simulation Steps")
        ax.set_ylabel("Count")
        ax.legend()

plt.tight_layout()
plt.show()

# Plot the Final Dynamic Network
plt.figure(figsize=(10, 8))
nx.draw(G, pos=nx.spring_layout(G), node_color=[node_colors[n] for n in G.nodes()], node_size=[node_sizes[n] for n in G.nodes()], edge_color="lightgray", with_labels=False)
plt.title("Final State of the Dynamic Network")
plt.show()
