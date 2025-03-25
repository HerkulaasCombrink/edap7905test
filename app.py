import streamlit as st
import pandas as pd
import numpy as np
import random
import pydeck as pdk
import time

st.set_page_config(page_title="Gauteng Territory Simulation", layout="wide")
st.title("🏙️ Agent-Based Simulation: Gauteng Province")

# --- Constants ---
NUM_TERRITORIES = 500
AGENT_COLORS = {'Agent A': [255, 0, 0], 'Agent B': [0, 0, 255], 'Agent C': [0, 255, 0]}  # RGB
AGENT_NAMES = list(AGENT_COLORS.keys())
GAUTENG_BOUNDS = {
    'lat_min': -26.7,
    'lat_max': -25.5,
    'lon_min': 27.5,
    'lon_max': 28.7
}

# --- Generate Random Territory Points in Gauteng ---
def generate_gauteng_territories(n=500):
    lats = np.random.uniform(low=GAUTENG_BOUNDS['lat_min'], high=GAUTENG_BOUNDS['lat_max'], size=n)
    lons = np.random.uniform(low=GAUTENG_BOUNDS['lon_min'], high=GAUTENG_BOUNDS['lon_max'], size=n)
    names = [f"G{i+1}" for i in range(n)]
    return pd.DataFrame({'name': names, 'lat': lats, 'lon': lons, 'owner': [None]*n})

# --- Initialize Session State ---
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'territories' not in st.session_state:
    st.session_state.territories = None
if 'agents' not in st.session_state:
    st.session_state.agents = None
if 'step' not in st.session_state:
    st.session_state.step = 0

# --- Core Functions ---
def get_adjacent(current_idx):
    return [i for i in range(NUM_TERRITORIES) if i != current_idx]

def reset_simulation():
    territories = generate_gauteng_territories(NUM_TERRITORIES)
    start_idxs = random.sample(range(NUM_TERRITORIES), 3)
    agents = {
        AGENT_NAMES[i]: {
            'color': AGENT_COLORS[AGENT_NAMES[i]],
            'location': start_idxs[i],
            'owned': set(),
            'cooldown': 0
        }
        for i in range(3)
    }
    st.session_state.territories = territories
    st.session_state.agents = agents
    st.session_state.step = 0
    st.session_state.simulation_started = True

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

    for target_idx, contenders in move_attempts.items():
        winner = random.choice(contenders) if len(contenders) > 1 else contenders[0]
        previous_owner = territories.at[target_idx, 'owner']
        territories.at[target_idx, 'owner'] = winner
        agents[winner]['location'] = target_idx
        agents[winner]['owned'].add(target_idx)

        if previous_owner and previous_owner != winner:
            agents[winner]['cooldown'] = 1

    st.session_state.step += 1

def render_map():
    territories = st.session_state.territories
    agents = st.session_state.agents

    layer_data = []
    for idx, row in territories.iterrows():
        owner = row['owner']
        color = agents[owner]['color'] if owner else [180, 180, 180]  # grey if unclaimed
        layer_data.append({
            'lat': row['lat'],
            'lon': row['lon'],
            'color': color,
            'name': row['name'],
            'owner': owner or "Unclaimed"
        })

    df_map = pd.DataFrame(layer_data)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=100,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=-26.1,
        longitude=28.2,
        zoom=8.5,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nOwner: {owner}"}
    ))

# --- Sidebar Controls ---
st.sidebar.markdown("### Simulation Controls")

if st.sidebar.button("🚀 Start Simulation"):
    reset_simulation()

if st.session_state.simulation_started:
    auto_steps = st.sidebar.slider("Number of seconds/steps", 1, 100, 10)

    if st.sidebar.button("▶️ Run Automatically"):
        for _ in range(auto_steps):
            run_step()
            render_map()
            time.sleep(1)
            st.experimental_rerun()

    if st.sidebar.button("⏭️ Next Step"):
        run_step()

    st.sidebar.write(f"Step: **{st.session_state.step}**")

# --- Main Content ---
if st.session_state.simulation_started:
    render_map()

    st.subheader("📊 Territories Owned")
    stats = {
        agent: len(data['owned'])
        for agent, data in st.session_state.agents.items()
    }
    st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Territories']).rename_axis("Agent"))
else:
    st.info("Click **Start Simulation** in the sidebar to begin.")
