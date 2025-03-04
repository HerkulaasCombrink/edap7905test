import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Title of the App
st.title("A/B Testing Click Demo")
st.write("Click to simulate user actions and compare two versions (A & B).")

# Initialize session state
if "clicks_A2" not in st.session_state:
    st.session_state.clicks_A2 = 0
    st.session_state.clicks_B2 = 0
    st.session_state.visitors_A = 0
    st.session_state.visitors_B = 0

# User Click Simulation
st.subheader("Click Simulation")
col1, col2 = st.columns(2)

with col1:
    if st.button("Click Version A"):
        st.session_state.clicks_A2 += 1
    st.session_state.visitors_A += 1
    st.write(f"Clicks: {st.session_state.clicks_A2}")
    st.write(f"Visitors: {st.session_state.visitors_A}")

with col2:
    if st.button("Click Version B"):
        st.session_state.clicks_B2 += 1
    st.session_state.visitors_B += 1
    st.write(f"Clicks: {st.session_state.clicks_B2}")
    st.write(f"Visitors: {st.session_state.visitors_B}")

# Compute conversion rates
rate_A = st.session_state.clicks_A2 / max(1, st.session_state.visitors_A)
rate_B = st.session_state.clicks_B2 / max(1, st.session_state.visitors_B)

def evaluate_ab_test(n_A, conv_A, n_B, conv_B):
    # Perform a two-proportion z-test
    p_A = conv_A / max(1, n_A)
    p_B = conv_B / max(1, n_B)
    p_pool = (conv_A + conv_B) / max(1, n_A + n_B)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/max(1, n_A) + 1/max(1, n_B)))
    z_score = (p_B - p_A) / max(1e-6, se)
    p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test
    return p_A, p_B, z_score, p_value

# Perform A/B Test Analysis
p_A, p_B, z_score, p_value = evaluate_ab_test(
    st.session_state.visitors_A, st.session_state.clicks_A2, 
    st.session_state.visitors_B, st.session_state.clicks_B2
)

# Display Results
st.subheader("Results")
st.write(f"**Conversion Rate for A:** {p_A:.2%}")
st.write(f"**Conversion Rate for B:** {p_B:.2%}")
st.write(f"**Z-Score:** {z_score:.2f}")
st.write(f"**P-Value:** {p_value:.4f}")

# Interpretation (Updates After Each Click)
st.subheader("Conclusion")
if p_value < 0.05:
    st.success("🚀 Version B performs significantly better than Version A! ✅ Keep using it!")
elif p_value > 0.95:
    st.error("⚠️ Version A is significantly better than Version B! Consider reverting.")
elif st.session_state.visitors_A + st.session_state.visitors_B < 30:
    st.info("📊 More data is needed to determine a clear winner. Keep testing!")
else:
    st.warning("No significant difference between A and B. Continue monitoring.")

# Visualization
fig, ax = plt.subplots()
bars = ax.bar(["Version A", "Version B"], [p_A, p_B], color=['blue', 'red'])
ax.bar_label(bars, fmt='%.2f%%')
ax.set_ylabel("Conversion Rate")
st.pyplot(fig)

# Title of the App
st.title("A/B/C Testing Click Demo")
st.write("Click to simulate user actions and compare three versions (A, B, & C).")

# Initialize session state
if "clicks_A2" not in st.session_state:
    st.session_state.clicks_A2 = 0
    st.session_state.clicks_B2 = 0
    st.session_state.clicks_C = 0
    st.session_state.visitors_A = 0
    st.session_state.visitors_B = 0
    st.session_state.visitors_C = 0

# User Click Simulation
st.subheader("Click Simulation")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Click Version A"):
        st.session_state.clicks_A2 += 1
    st.session_state.visitors_A += 1
    st.write(f"Clicks: {st.session_state.clicks_A2}")
    st.write(f"Visitors: {st.session_state.visitors_A}")

with col2:
    if st.button("Click Version B"):
        st.session_state.clicks_B2 += 1
    st.session_state.visitors_B += 1
    st.write(f"Clicks: {st.session_state.clicks_B2}")
    st.write(f"Visitors: {st.session_state.visitors_B}")

with col3:
    if st.button("Click Version C"):
        st.session_state.clicks_C += 1
    st.session_state.visitors_C += 1
    st.write(f"Clicks: {st.session_state.clicks_C}")
    st.write(f"Visitors: {st.session_state.visitors_C}")

# Compute conversion rates
rate_A = st.session_state.clicks_A2 / max(1, st.session_state.visitors_A)
rate_B = st.session_state.clicks_B2 / max(1, st.session_state.visitors_B)
rate_C = st.session_state.clicks_C / max(1, st.session_state.visitors_C)

def evaluate_abc_test(n_A, conv_A, n_B, conv_B, n_C, conv_C):
    # Perform a three-proportion z-test (pairwise comparisons)
    p_A = conv_A / max(1, n_A)
    p_B = conv_B / max(1, n_B)
    p_C = conv_C / max(1, n_C)
    
    p_pool = (conv_A + conv_B + conv_C) / max(1, n_A + n_B + n_C)
    se_A_B = np.sqrt(p_pool * (1 - p_pool) * (1/max(1, n_A) + 1/max(1, n_B)))
    se_A_C = np.sqrt(p_pool * (1 - p_pool) * (1/max(1, n_A) + 1/max(1, n_C)))
    se_B_C = np.sqrt(p_pool * (1 - p_pool) * (1/max(1, n_B) + 1/max(1, n_C)))
    
    z_AB = (p_B - p_A) / max(1e-6, se_A_B)
    z_AC = (p_C - p_A) / max(1e-6, se_A_C)
    z_BC = (p_C - p_B) / max(1e-6, se_B_C)
    
    p_value_AB = 1 - stats.norm.cdf(z_AB)  # One-tailed test
    p_value_AC = 1 - stats.norm.cdf(z_AC)
    p_value_BC = 1 - stats.norm.cdf(z_BC)
    
    return p_A, p_B, p_C, z_AB, z_AC, z_BC, p_value_AB, p_value_AC, p_value_BC

# Perform A/B/C Test Analysis
p_A, p_B, p_C, z_AB, z_AC, z_BC, p_value_AB, p_value_AC, p_value_BC = evaluate_abc_test(
    st.session_state.visitors_A, st.session_state.clicks_A2, 
    st.session_state.visitors_B, st.session_state.clicks_B2,
    st.session_state.visitors_C, st.session_state.clicks_C
)

# Display Results
st.subheader("Results")
st.write(f"**Conversion Rate for A:** {p_A:.2%}")
st.write(f"**Conversion Rate for B:** {p_B:.2%}")
st.write(f"**Conversion Rate for C:** {p_C:.2%}")
st.write(f"**Z-Score (A vs B):** {z_AB:.2f}, **P-Value:** {p_value_AB:.4f}")
st.write(f"**Z-Score (A vs C):** {z_AC:.2f}, **P-Value:** {p_value_AC:.4f}")
st.write(f"**Z-Score (B vs C):** {z_BC:.2f}, **P-Value:** {p_value_BC:.4f}")

# Interpretation (Updates After Each Click)
st.subheader("Conclusion")
if min(p_value_AB, p_value_AC, p_value_BC) < 0.05:
    best_version = ["A", "B", "C"][np.argmax([p_A, p_B, p_C])]
    st.success(f"🚀 Version {best_version} performs significantly better! ✅ Keep using it!")
elif st.session_state.visitors_A + st.session_state.visitors_B + st.session_state.visitors_C < 30:
    st.info("📊 More data is needed to determine a clear winner. Keep testing!")
else:
    st.warning("No significant difference between A, B, and C. Continue monitoring.")

# Visualization
fig, ax = plt.subplots()
bars = ax.bar(["Version A", "Version B", "Version C"], [p_A, p_B, p_C], color=['blue', 'red', 'green'])
ax.bar_label(bars, fmt='%.2f%%')
ax.set_ylabel("Conversion Rate")
st.pyplot(fig)
