# streamlit_app.py

import streamlit as st
import pandas as pd
import random
import numpy as np

# Set page config
st.set_page_config(page_title="Agent Territory Competition", layout="wide")

st.title("🇿🇦 Agent-Based Territory Simulation: South Africa")

# Initialize territory points in South Africa (simplified)
territories = pd.DataFrame({
    'lat': [-33.918861, -26.204103, -29.858680, -25.747868, -28.4793, -24.6544, -30.5595],
    'lon': [18.423300, 28.047305, 31.021840, 28.229271, 24.6727, 25.9086, 22.9375],
    'name': ['Cape Town', 'Johannesburg', 'Durban', 'Pretoria', 'Kimberley', 'Polokwane', 'Upington'],
    'owner': [None]*7
})

# Sidebar for number of steps
steps = st.sidebar.slider("Number of steps (episodes)", min_value=1, max_value=50, value=10)

# Initialize agents
colors = ['red', 'blue', 'green']
agents = {
    'Agent A': {'color': 'red', 'location': 0, 'owned': set(), 'cooldown': 0},
    'Agent B': {'color': 'blue', 'location': 1, 'owned': set(), 'cooldown': 0},
    'Agent C': {'color': 'green', 'location': 2, 'owned': set(), 'cooldown': 0},
}

def get_adjacent(current_idx):
    """Simplified version of adjacency: just allow moving to any other territory"""
    return [i for i in range(len(territories)) if i != current_idx]

# Track move attempts
for step in range(steps):
    move_attempts = {}
    
    for agent_name, agent in agents.items():
        if agent['cooldown'] > 0:
            agent['cooldown'] -= 1
            continue  # skip this turn
        
        current_idx = agent['location']
        options = get_adjacent(current_idx)

        # Prefer unclaimed territories
        unclaimed = [i for i in options if territories.at[i, 'owner'] is None]
        if unclaimed:
            target_idx = random.choice(unclaimed)
        else:
            # Steal randomly if no unclaimed
            target_idx = random.choice(options)

        if target_idx not in move_attempts:
            move_attempts[target_idx] = []
        move_attempts[target_idx].append(agent_name)

    # Resolve conflicts and apply moves
    for target_idx, contenders in move_attempts.items():
        if len(contenders) == 1:
            winner = contenders[0]
        else:
            winner = random.choice(contenders)

        for contender in contenders:
            if contender != winner:
                continue  # they failed the move
            
        previous_owner = territories.at[target_idx, 'owner']
        territories.at[target_idx, 'owner'] = winner
        agents[winner]['location'] = target_idx
        agents[winner]['owned'].add(target_idx)

        if previous_owner and previous_owner != winner:
            agents[winner]['cooldown'] = 1  # penalty for stealing

# Display map
map_data = pd.DataFrame(columns=['lat', 'lon', 'color'])

for agent_name, agent in agents.items():
    for idx in agent['owned']:
        row = territories.iloc[idx]
        map_data = pd.concat([map_data, pd.DataFrame([{
            'lat': row['lat'],
            'lon': row['lon'],
            'color': agent['color']
        }])])

st.map(map_data[['lat', 'lon']])

# Show stats
st.subheader("Territories Owned")
stats = {agent: len(data['owned']) for agent, data in agents.items()}
st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Territories']).rename_axis("Agent"))

