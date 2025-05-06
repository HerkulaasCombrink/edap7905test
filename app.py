import streamlit as st
import random
import pandas as pd
import altair as alt

st.set_page_config(page_title="Lotto Number Predictor", layout="centered")
st.title("🎲 Lotto Number Predictor with Range Bubble Visualization")
st.markdown("""
Predict 5 numbers from 1–49 and a bonus number from 1–20.
This strategy excludes obvious sequences (e.g., 1,2,3,4,5 or constant-step sequences like 3,6,9,12,15),
and visualises each chosen number alongside its ±3 range on a continuum.
""" )

# Utility to detect arithmetic sequences
def is_arithmetic_sequence(nums):
    if len(nums) < 2:
        return False
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    return all(d == diffs[0] for d in diffs)

# Generate one lotto combination
def generate_lotto_numbers():
    available = list(range(1, 50))
    chosen = []

    # Pick 5 main numbers
    for _ in range(5):
        num = random.choice(available)
        chosen.append(num)
        available.remove(num)

    chosen_sorted = sorted(chosen)
    # Exclude trivial patterns
    if chosen_sorted == [1,2,3,4,5] or is_arithmetic_sequence(chosen_sorted):
        return generate_lotto_numbers()

    # Pick bonus number
    bonus = random.choice(list(range(1, 21)))
    return chosen_sorted, bonus

# Sidebar controls
num_combos = st.sidebar.slider("Number of combinations to generate", min_value=1, max_value=10, value=1)

if st.button("Generate Lotto Numbers"):
    for idx in range(num_combos):
        combo, bonus = generate_lotto_numbers()
        exp = st.expander(f"Combination {idx+1}")
        with exp:
            st.subheader(f"Numbers: {combo}  •  Bonus: {bonus}")

            # Compute ranges for each main number
            ranges = []
            for n in combo:
                start = max(1, n - 3)
                end = min(49, n + 3)
                ranges.append({"Number": n, "Range Start": start, "Range End": end})

            range_df = pd.DataFrame(ranges)
            st.write("**Number Ranges (±3)**")
            st.table(range_df)

            # Bubble + rule visualization over continuum
            rule = alt.Chart(range_df).mark_rule(color='gray', size=4).encode(
                x='Range Start:Q',
                x2='Range End:Q'
            )
            circles = alt.Chart(range_df).mark_circle(size=100).encode(
                x='Number:Q',
                tooltip=['Number', 'Range Start', 'Range End']
            )
            chart = alt.layer(rule, circles).properties(
                width=600,
                height=100,
                title='Main Number Ranges'
            )
            st.altair_chart(chart, use_container_width=True)
