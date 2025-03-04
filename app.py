import streamlit as st
import numpy as np
import scipy.stats as stats

# Title of the App
st.title("A/B Testing Demo")
st.write("Compare two versions (A & B) and analyze the better performer.")

# User Inputs for conversion data
st.sidebar.header("User Input: Conversions")
n_A = st.sidebar.number_input("Visitors for A:", min_value=1, value=1000)
conv_A = st.sidebar.number_input("Conversions for A:", min_value=0, max_value=n_A, value=50)
n_B = st.sidebar.number_input("Visitors for B:", min_value=1, value=1000)
conv_B = st.sidebar.number_input("Conversions for B:", min_value=0, max_value=n_B, value=70)

# Compute conversion rates
rate_A = conv_A / n_A
rate_B = conv_B / n_B

def evaluate_ab_test(n_A, conv_A, n_B, conv_B):
    # Perform a two-proportion z-test
    p_A = conv_A / n_A
    p_B = conv_B / n_B
    p_pool = (conv_A + conv_B) / (n_A + n_B)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
    z_score = (p_B - p_A) / se
    p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test
    return p_A, p_B, z_score, p_value

# Perform A/B Test Analysis
p_A, p_B, z_score, p_value = evaluate_ab_test(n_A, conv_A, n_B, conv_B)

# Display Results
st.subheader("Results")
st.write(f"**Conversion Rate for A:** {p_A:.2%}")
st.write(f"**Conversion Rate for B:** {p_B:.2%}")
st.write(f"**Z-Score:** {z_score:.2f}")
st.write(f"**P-Value:** {p_value:.4f}")

# Interpretation
st.subheader("Conclusion")
if p_value < 0.05:
    st.success("Version B performs significantly better than Version A! ✅")
else:
    st.warning("No significant difference between A and B. More data may be needed.")

# Visualization
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
bars = ax.bar(["Version A", "Version B"], [p_A, p_B], color=['blue', 'red'])
ax.bar_label(bars, fmt='%.2f%%')
ax.set_ylabel("Conversion Rate")
st.pyplot(fig)
