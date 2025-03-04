import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the App
st.title("Epsilon-Greedy Multi-Armed Bandit")
st.write("Click to simulate user actions and let the algorithm dynamically learn which version (A or B) is better.")

# Parameters
EPSILON = 0.1  # Exploration rate

# Initialize session state for MAB
if "rewards_A" not in st.session_state:
    st.session_state.rewards_A = 0
    st.session_state.rewards_B = 0
    st.session_state.visits_A = 0
    st.session_state.visits_B = 0

# Choose action based on epsilon-greedy strategy
def choose_action():
    if np.random.rand() < EPSILON:
        return np.random.choice(["A", "B"])  # Explore
    else:
        return "A" if (st.session_state.rewards_A / max(1, st.session_state.visits_A)) > \
                     (st.session_state.rewards_B / max(1, st.session_state.visits_B)) else "B"  # Exploit

# User Click Simulation
st.subheader("Click Simulation")
col1, col2 = st.columns(2)

with col1:
    if st.button("Click Version A"):
        st.session_state.visits_A += 1
        if np.random.rand() < 0.5:  # Simulated reward probability for A
            st.session_state.rewards_A += 1
    st.write(f"Clicks: {st.session_state.rewards_A}")
    st.write(f"Visitors: {st.session_state.visits_A}")

with col2:
    if st.button("Click Version B"):
        st.session_state.visits_B += 1
        if np.random.rand() < 0.6:  # Simulated reward probability for B
            st.session_state.rewards_B += 1
    st.write(f"Clicks: {st.session_state.rewards_B}")
    st.write(f"Visitors: {st.session_state.visits_B}")

# Compute estimated conversion rates
rate_A = st.session_state.rewards_A / max(1, st.session_state.visits_A)
rate_B = st.session_state.rewards_B / max(1, st.session_state.visits_B)

# Display Results
st.subheader("Results")
st.write(f"**Estimated Conversion Rate for A:** {rate_A:.2%}")
st.write(f"**Estimated Conversion Rate for B:** {rate_B:.2%}")

# Interpretation
st.subheader("Conclusion")
if st.session_state.visits_A + st.session_state.visits_B < 30:
    st.info("📊 More data is needed to determine a clear winner. Keep testing!")
elif rate_A > rate_B:
    st.success("🚀 Version A appears to be better! ✅ Exploiting A more.")
elif rate_B > rate_A:
    st.success("🚀 Version B appears to be better! ✅ Exploiting B more.")
else:
    st.warning("No significant difference yet. Continue exploring.")

# Visualization
fig, ax = plt.subplots()
bars = ax.bar(["Version A", "Version B"], [rate_A, rate_B], color=['blue', 'red'])
ax.bar_label(bars, fmt='%.2f%%')
ax.set_ylabel("Estimated Conversion Rate")
st.pyplot(fig)
