import streamlit as st
import random
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh

# --- Config ---
st.set_page_config(page_title="Athlete Simulation Dashboard", layout="wide")
st.title("🏃 Athlete Damage & Fatigue Dashboard (Synthetic Only)")

# --- Session State Init ---
if "sim_state" not in st.session_state:
    st.session_state.sim_state = {
        "running": False,
        "current_time": 0,
        "fatigue": 0.0,
        "cognition": 100.0,
        "fatigue_log": [],
        "cognition_log": [],
        "time_log": [],
        "damage_log": pd.DataFrame(columns=["Time", "Body Part", "Fatigue Δ", "Cognition Δ"]),
    }

# --- Damage Profiles ---
BODY_PART_IMPACTS = {
    "head": {"fatigue": 10, "cognition": 20},
    "chest": {"fatigue": 8, "cognition": 10},
    "arm": {"fatigue": 5, "cognition": 3},
    "leg": {"fatigue": 5, "cognition": 3},
    "abdomen": {"fatigue": 3, "cognition": 1},
}

# --- Controls ---
with st.sidebar:
    st.header("Simulation Controls")
    if st.button("▶ Start / Resume"):
        st.session_state.sim_state["running"] = True
    if st.button("⏸ Pause"):
        st.session_state.sim_state["running"] = False
    if st.button("🔁 Reset"):
        st.session_state.sim_state = {
            "running": False,
            "current_time": 0,
            "fatigue": 0.0,
            "cognition": 100.0,
            "fatigue_log": [],
            "cognition_log": [],
            "time_log": [],
            "damage_log": pd.DataFrame(columns=["Time", "Body Part", "Fatigue Δ", "Cognition Δ"]),
        }

    st.write("⏱️ Time:", st.session_state.sim_state["current_time"], "s")

# --- Simulation Update ---
if st.session_state.sim_state["running"]:
    if st.session_state.sim_state["current_time"] < 1000:
        sim = st.session_state.sim_state
        sim["current_time"] += 1
        sim["fatigue"] += 0.02
        sim["cognition"] -= 0.015

        # Random damage event
        if random.random() < 0.1:
            part = random.choice(list(BODY_PART_IMPACTS.keys()))
            dmg = BODY_PART_IMPACTS[part]
            sim["fatigue"] += dmg["fatigue"]
            sim["cognition"] -= dmg["cognition"]
            new_entry = {
                "Time": sim["current_time"],
                "Body Part": part,
                "Fatigue Δ": dmg["fatigue"],
                "Cognition Δ": -dmg["cognition"]
            }
            sim["damage_log"] = pd.concat([sim["damage_log"], pd.DataFrame([new_entry])], ignore_index=True)

        sim["fatigue_log"].append(sim["fatigue"])
        sim["cognition_log"].append(sim["cognition"])
        sim["time_log"].append(sim["current_time"])

        # Optional: prevent memory overflow
        if len(sim["time_log"]) > 1000:
            sim["fatigue_log"] = sim["fatigue_log"][-1000:]
            sim["cognition_log"] = sim["cognition_log"][-1000:]
            sim["time_log"] = sim["time_log"][-1000:]

        # SAFE AUTO-REFRESH every 1s
        st_autorefresh(interval=1000, limit=1000, key="auto-refresh")
    else:
        st.warning("🏁 Simulation completed.")
        st.session_state.sim_state["running"] = False

# --- Dashboard Layout ---
col1, col2 = st.columns([1.5, 1])

# --- Time-series Chart ---
with col1:
    st.subheader("📈 Fatigue & Cognition Over Time")
    df = pd.DataFrame({
        "Time": st.session_state.sim_state["time_log"],
        "Fatigue": st.session_state.sim_state["fatigue_log"],
        "Cognition": st.session_state.sim_state["cognition_log"],
    })
    if not df.empty:
        line_chart = alt.Chart(df).transform_fold(
            ["Fatigue", "Cognition"], as_=["Metric", "Value"]
        ).mark_line().encode(
            x="Time:Q",
            y="Value:Q",
            color="Metric:N"
        ).properties(height=300)
        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("Simulation not started yet.")

# --- Status Metrics ---
with col2:
    st.subheader("📊 Status")
    fatigue = st.session_state.sim_state["fatigue"]
    cognition = st.session_state.sim_state["cognition"]
    performance = round(100 - ((fatigue + (100 - cognition)) / 2), 2)

    st.metric("Fatigue", f"{fatigue:.2f}")
    st.metric("Cognition", f"{cognition:.2f}")
    st.metric("Performance", f"{performance:.2f}")

    if fatigue >= 90 or cognition <= 40:
        st.error("🔴 Replace Athlete! Threshold passed.")
    elif fatigue >= 70 or cognition <= 60:
        st.warning("🟠 Break Required!")

# --- Damage Log Table ---
st.subheader("📄 Synthetic Damage Log")
st.dataframe(st.session_state.sim_state["damage_log"], use_container_width=True)
