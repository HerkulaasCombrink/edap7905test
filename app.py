import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import networkx as nx

# Initialize simulation parameters
def get_model_params():
    return {
        "N": st.sidebar.slider("Number of agents", 50, 500, 100),
        "initial_infected": st.sidebar.slider("Initial Number of Infected", 1, 10, 3),
        "infection_probability": st.sidebar.slider("Infection Probability", 0.0, 1.0, 0.5),
        "steps": 10,  # Running for 10 seconds
    }

# Agent class
class Agent:
    def __init__(self, unique_id, status, size):
        self.unique_id = unique_id
        self.status = status  # "infected" or "susceptible"
        self.size = size  # Determines susceptibility
        self.infection_timer = 0  # Timer for conversion delay

    def interact(self, neighbors, infection_probability):
        if self.status == "infected":
            for neighbor in neighbors:
                if neighbor.status == "susceptible":
                    susceptibility_factor = 1.0 / neighbor.size  # Smaller nodes are more susceptible
                    if random.random() < (infection_probability * susceptibility_factor):
                        neighbor.infection_timer = self.size  # Delay based on size

    def update_status(self):
        if self.status == "susceptible" and self.infection_timer > 0:
            self.infection_timer -= 1
            if self.infection_timer == 0:
                self.status = "infected"

# Disease Spread Model
class DiseaseSpreadModel:
    def __init__(self, **params):
        self.num_agents = params["N"]
        self.infection_probability = params["infection_probability"]
        self.G = nx.barabasi_albert_graph(self.num_agents, 3)
        self.agents = {}
        
        all_nodes = list(self.G.nodes())
        initial_infected = random.sample(all_nodes, params["initial_infected"])  # Select initial infected nodes
        
        for node in all_nodes:
            size = random.choice([1, 2, 3, 4])  # 1 (most susceptible) to 4 (least susceptible)
            status = "infected" if node in initial_infected else "susceptible"
            self.agents[node] = Agent(node, status, size)
        
        # Ensure all nodes start as gray (susceptible), except the specified infected nodes
        for node in all_nodes:
            if node not in initial_infected:
                self.agents[node].status = "susceptible"
            else:
                self.agents[node].status = "infected"
        
        self.node_positions = nx.spring_layout(self.G)  # Fix network shape
        self.history = []

    def step(self, step_num):
        for node, agent in self.agents.items():
            neighbors = [self.agents[n] for n in self.G.neighbors(node)]
            agent.interact(neighbors, self.infection_probability)
        
        for agent in self.agents.values():
            agent.update_status()
        
        self.history.append({node: agent.status for node, agent in self.agents.items()})

# Visualization function
def plot_visuals(G, agents, positions):
    color_map = {"infected": "red", "susceptible": "gray"}
    node_colors = [color_map[agents[node].status] for node in G.nodes()]
    node_sizes = [agents[node].size * 50 for node in G.nodes()]  # Adjust node size by susceptibility
    
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos=positions, ax=ax, node_color=node_colors, with_labels=False, node_size=node_sizes, edge_color="gray")
    ax.set_title("Disease Spread Network")
    return fig

# Streamlit App
st.title("Scale-Free Network Disease Spread Simulation")
params = get_model_params()

if st.button("Run Simulation"):
    model = DiseaseSpreadModel(**params)
    progress_bar = st.progress(0)
    visual_plot = st.empty()
    
    for step_num in range(1, params["steps"] + 1):
        model.step(step_num)
        progress_bar.progress(step_num / params["steps"])
        fig = plot_visuals(model.G, model.agents, model.node_positions)
        visual_plot.pyplot(fig)
    
    st.write("Simulation Complete.")
    
    st.write("Simulation Complete.")
if st.button("Test this"):
  time_series = np.random.randn(100)
