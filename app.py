import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pygame

st.header("This is mine")
# Generate random time series data
if st.button("Test this"):
    def get_model_params():
    return {
        "N": st.sidebar.slider("Number of agents", 50, 500, 100),
        "misinformation_spread_prob": st.sidebar.slider("Misinformation Spread Probability", 0.0, 1.0, 0.3),
        "skeptic_ratio": st.sidebar.slider("Skeptic Ratio", 0.0, 1.0, 0.2),
        "influencer_ratio": st.sidebar.slider("Influencer Ratio", 0.0, 1.0, 0.1),
        "fact_check_prob": st.sidebar.slider("Fact Check Probability", 0.0, 1.0, 0.1),
        "epsilon": st.sidebar.slider("Epsilon (Exploration Rate)", 0.0, 1.0, 0.1),
        "steps": 200,
        "duration": 100  # Duration of simulation in seconds
    }

# Agent class
class Agent:
    def __init__(self, unique_id, belief_status, epsilon):
        self.unique_id = unique_id
        self.belief_status = belief_status  # "believer", "skeptic", "neutral", "influencer"
        self.epsilon = epsilon  # Exploration-exploitation trade-off

    def interact(self, neighbors, misinformation_spread_prob, fact_check_prob):
        if random.random() < self.epsilon:
            self.belief_status = random.choice(["believer", "skeptic", "neutral", "influencer"])
        else:
            for neighbor in neighbors:
                if self.belief_status == "believer" and neighbor.belief_status == "neutral":
                    if np.random.random() < misinformation_spread_prob:
                        neighbor.belief_status = "believer"
                elif self.belief_status == "skeptic" and neighbor.belief_status == "believer":
                    if np.random.random() < fact_check_prob:
                        neighbor.belief_status = "neutral"

# Misinformation Model
class MisinformationModel:
    def __init__(self, **params):
        self.num_agents = params["N"]
        self.misinformation_spread_prob = params["misinformation_spread_prob"]
        self.fact_check_prob = params["fact_check_prob"]
        self.skeptic_ratio = params["skeptic_ratio"]
        self.epsilon = params["epsilon"]
        self.G = nx.barabasi_albert_graph(self.num_agents, 3)
        self.agents = {}

        for i, node in enumerate(self.G.nodes()):
            belief_status = np.random.choice(["believer", "skeptic", "neutral", "influencer"],
                                             p=[0.4, self.skeptic_ratio, 0.4, params["influencer_ratio"]])
            self.agents[node] = Agent(i, belief_status, self.epsilon)

        self.history = []
        self.interaction_counts = []

    def step(self, step_num):
        new_state = {agent_id: agent.belief_status for agent_id, agent in self.agents.items()}
        self.history.append(new_state)
        interactions = 0

        for node, agent in self.agents.items():
            neighbors = [self.agents[n] for n in self.G.neighbors(node)]
            prev_state = agent.belief_status
            agent.interact(neighbors, self.misinformation_spread_prob, self.fact_check_prob)
            if prev_state != agent.belief_status:
                interactions += 1
        
        self.interaction_counts.append(interactions)

# Visualization function
def plot_interactions(model):
    fig, ax = plt.subplots()
    ax.plot(model.interaction_counts, label="Misinformation Interactions Over Time", color="red")
    ax.set_xlabel("Simulation Steps")
    ax.set_ylabel("Number of Interactions")
    ax.set_title("Tracking Misinformation Spread")
    ax.legend()
    st.pyplot(fig)

# Pygame bouncing balls simulation
def bouncing_balls_simulation(duration=100):
    pygame.init()
    width, height = 500, 500
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    
    balls = [{
        "pos": [random.randint(20, 480), random.randint(20, 480)],
        "vel": [random.choice([-2, 2]), random.choice([-2, 2])],
        "color": random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
    } for _ in range(30)]
    
    start_time = time.time()
    while time.time() - start_time < duration:
        screen.fill((0, 0, 0))
        for ball in balls:
            ball["pos"][0] += ball["vel"][0]
            ball["pos"][1] += ball["vel"][1]
            
            if ball["pos"][0] <= 10 or ball["pos"][0] >= 490:
                ball["vel"][0] *= -1
            if ball["pos"][1] <= 10 or ball["pos"][1] >= 490:
                ball["vel"][1] *= -1
            
            pygame.draw.circle(screen, ball["color"], ball["pos"], 10)
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

# Streamlit App
st.title("Scale-Free Agent-Based Misinformation Simulation")
params = get_model_params()

if st.button("Run Simulation"):
    model = MisinformationModel(**params)
    for step_num in range(1, params["steps"] + 1):
        model.step(step_num)
    plot_interactions(model)
    st.write("Starting Bouncing Balls Visualization...")
    bouncing_balls_simulation(params["duration"])
