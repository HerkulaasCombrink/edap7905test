import streamlit as st
import pandas as pd
import random
import numpy as np

st.set_page_config(page_title="Agent Territory Simulation", layout="wide")
st.title("🇿🇦 Step-by-Step Agent-Based Territory Simulation")

# --- Constants ---
NUM_TERRITORIES = 72
AGENT_COLORS = {'Agent A': 'red', 'Agent B': 'blue', 'Agent C': 'green'}

# --- Helper: Generate Random Territory Points in SA ---
def generate_territories(n=72):
    lats = np.random.uniform(low=-35.0, high=-22.0, size=n)
    lons = np.random.uniform(low=16.0, high=33.0, size=n)
    names = [f"T{i+1}" for i in range(n)]
    return pd.DataFrame({'name': names, 'lat': lats, 'lon': lons, 'owner': [None]*n})

# --- Session State Initialization ---
if 'territories' not in st.session_state:
    st.session_state.territories = generate_territories(NUM_TERRITORIES)

if 'agents' not in st.session_state:
    # Random starting positions
    start_idxs = random.sample(range(NUM_TERRITORIES), 3)
    st.session_state.agents = {
        'Agent A': {'color': 'red', 'location': start_idxs[0], 'owned': set(), 'cooldown': 0},
        'Agent B': {'color': 'blue', 'location': start_idxs[1], 'owned': set(), 'cooldown': 0},
        'Agent C': {'color': 'green', 'location': start_idxs[2], 'owned': set(), 'cooldown': 0},
    }

if 'step' not in st.session_state:
    st.session_state.step = 0

# --- Helpers ---
def get_adjacent(current_idx):
    # Simplified adjacency — allow movement to any other territory
    return [i for i in range(NUM_TERRITORIES) if i != current_idx]

def reset_simulation():
    st.session_state.territories = generate_territories(NUM_TERRITORIES)
    start_idxs = random.sample(range(NUM_TERRITORIES), 3)
    st.session_state.agents = {
        'Agent A': {'color': 'red', 'location': start_idxs[0], 'owned': set(), 'cooldown': 0},
        'Agent B': {'color': 'blue', 'location': start_idxs[1], 'owned': set(), 'cooldown': 0},
        'Agent C': {'color': 'green', 'location': start_idxs[2], 'owned': set(), 'cooldown': 0},
    }
    st.session_state.step = 0

def run_step():
    move_attempts = {}
    territories = st.session_state.territories
    agents = st.session_state.agents

    for agent_name, agent in agents.items():
        if agent['cooldown'] > 0:
            agent['cooldown'] -= 1
            continue

        current_idx = agent['location']
        options = get_adjacent(current_idx)

        unclaimed = [i for i in options if territories.at[i, 'owner'] is None]
        if unclaimed:
            target_idx = random.choice(unclaimed)
        else:
            target_idx = random.choice(options)

        if target_idx not in move_attempts:
            move_attempts[target_idx] = []
        move_attempts[target_idx].append(agent_name)

    # Resolve conflicts
    for target_idx, contenders in move_attempts.items():
        winner = random.choice(contenders) if len(contenders) > 1 else contenders[0]

        previous_owner = territories.at[target_idx, 'owner']
        territories.at[target_idx, 'owner'] = winner
        agents[winner]['location'] = target_idx
        agents[winner]['owned'].add(target_idx)

        if previous_owner and previous_owner != winner:
            agents[winner]['cooldown'] = 1  # penalty

    st.session_state.step += 1

# --- UI ---
st.sidebar.markdown("### Controls")
if st.sidebar.button("Start Simulation 🔁"):
    reset_simulation()

if st.sidebar.button("Next Step ⏭️"):
    run_step()

st.sidebar.write(f"Current Step: {st.session_state.step}")

# --- Map Visualization ---
map_data = pd.DataFrame(columns=['lat', 'lon', 'color'])
for agent_name, agent in st.session_state.agents.items():
    for idx in agent['owned']:
        row = st.session_state.territories.iloc[idx]
        map_data = pd.concat([map_data, pd.DataFrame([{
            'lat': row['lat'],
            'lon': row['lon'],
            'color': agent['color']
        }])])

if not map_data.empty:
    st.map(map_data[['lat', 'lon']])
else:
    st.info("Click 'Start Simulation' to begin.")

# --- Stats ---
st.subheader("📊 Territories Owned")
stats = {agent: len(data['owned']) for agent, data in st.session_state.agents.items()}
st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Territories']).rename_axis("Agent"))
