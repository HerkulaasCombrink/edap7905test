import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Title of the App
st.title("Multi-Armed Bandit: Epsilon-Greedy, Thompson Sampling, and UCB")
st.write("Click to simulate user actions and let different algorithms dynamically learn which version (A, B, C, or D) is better.")

# Parameters
EPSILON = 0.1  # Exploration rate for Epsilon-Greedy
ALPHA, BETA = 1, 1  # Parameters for Thompson Sampling
C = 2  # Exploration factor for UCB

# Initialize session state for MAB
if "rewards" not in st.session_state:
    st.session_state.rewards = {"A": 0, "B": 0, "C": 0, "D": 0}
    st.session_state.visits = {"A": 0, "B": 0, "C": 0, "D": 0}
    st.session_state.alpha = {"A": ALPHA, "B": ALPHA, "C": ALPHA, "D": ALPHA}
    st.session_state.beta = {"A": BETA, "B": BETA, "C": BETA, "D": BETA}

# Choose action based on epsilon-greedy strategy
def choose_action_epsilon_greedy():
    if np.random.rand() < EPSILON:
        return np.random.choice(["A", "B", "C", "D"])  # Explore
    else:
        return max(st.session_state.rewards, key=lambda k: st.session_state.rewards[k] / max(1, st.session_state.visits[k]))

# Choose action based on Upper Confidence Bound (UCB)
def choose_action_ucb():
    total_visits = sum(st.session_state.visits.values()) + 1
    ucb_values = {arm: (st.session_state.rewards[arm] / max(1, st.session_state.visits[arm])) + C * np.sqrt(np.log(total_visits) / max(1, st.session_state.visits[arm])) for arm in st.session_state.rewards}
    return max(ucb_values, key=ucb_values.get)

# Choose action based on Thompson Sampling
def choose_action_thompson():
    samples = {arm: np.random.beta(st.session_state.alpha[arm], st.session_state.beta[arm]) for arm in st.session_state.rewards}
    return max(samples, key=samples.get)

# User Click Simulation
st.subheader("Click Simulation")
cols = st.columns(4)
for idx, arm in enumerate(["A", "B", "C", "D"]):
    with cols[idx]:
        if st.button(f"Click Version {arm}"):
            st.session_state.visits[arm] += 1
            if np.random.rand() < 0.5:  # Simulated reward probability
                st.session_state.rewards[arm] += 1
                st.session_state.alpha[arm] += 1  # Update for Thompson Sampling
            else:
                st.session_state.beta[arm] += 1  # Update for Thompson Sampling
        st.write(f"Clicks: {st.session_state.rewards[arm]}")
        st.write(f"Visitors: {st.session_state.visits[arm]}")

# Compute estimated conversion rates
conversion_rates = {arm: st.session_state.rewards[arm] / max(1, st.session_state.visits[arm]) for arm in st.session_state.rewards}

# Display Results
st.subheader("Results")
for arm, rate in conversion_rates.items():
    st.write(f"**Estimated Conversion Rate for {arm}:** {rate:.2%}")

# Decision Making using MAB strategies
epsilon_choice = choose_action_epsilon_greedy()
ucb_choice = choose_action_ucb()
thompson_choice = choose_action_thompson()

# Interpretation
st.subheader("Conclusion")
if sum(st.session_state.visits.values()) < 30:
    st.info("📊 More data is needed to determine a clear winner. Keep testing!")
st.success(f"🚀 **Epsilon-Greedy prefers:** {epsilon_choice}")
st.success(f"🚀 **UCB prefers:** {ucb_choice}")
st.success(f"🚀 **Thompson Sampling prefers:** {thompson_choice}")

# Visualization
fig, ax = plt.subplots()
bars = ax.bar(["A", "B", "C", "D"], [conversion_rates["A"], conversion_rates["B"], conversion_rates["C"], conversion_rates["D"]], color=['blue', 'red', 'green', 'purple'])
ax.bar_label(bars, fmt='%.2f%%')
ax.set_ylabel("Estimated Conversion Rate")
st.pyplot(fig)
