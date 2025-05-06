import streamlit as st
import random
import pandas as pd
import altair as alt

st.set_page_config(page_title="Lotto Number Predictor", layout="centered")
st.title("🎲 Lotto Number Predictor with Range Visualization")
st.markdown("""
Predict 5 numbers from 1–49 and a bonus number from 1–20.
This strategy excludes obvious sequences (e.g., 1,2,3,4,5 or constant-step sequences like 3,6,9,12,15),
and visualises probability distributions and chosen-number ranges.
""" )

# Utility to detect arithmetic sequences
def is_arithmetic_sequence(nums):
    if len(nums) < 2:
        return False
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    return all(d == diffs[0] for d in diffs)

# Generate one lotto combination and capture probability frames
def generate_lotto_numbers():
    available = list(range(1, 50))
    chosen = []
    frames = []

    # Choose 5 main numbers
    for pick_idx in range(1, 6):
        prob = 1 / len(available)
        df = pd.DataFrame({
            "Number": available,
            "Probability": [prob] * len(available)
        })
        frames.append((f"Pick {pick_idx}", df))

        chosen_num = random.choices(available, weights=[prob]*len(available), k=1)[0]
        chosen.append(chosen_num)
        available.remove(chosen_num)

    chosen_sorted = sorted(chosen)
    # Exclude trivial patterns
    if chosen_sorted == [1,2,3,4,5] or is_arithmetic_sequence(chosen_sorted):
        return generate_lotto_numbers()

    # Bonus number
    bonus_avail = list(range(1, 21))
    bonus_df = pd.DataFrame({
        "Bonus Number": bonus_avail,
        "Probability": [1/len(bonus_avail)] * len(bonus_avail)
    })
    frames.append(("Bonus", bonus_df))
    bonus_num = random.choice(bonus_avail)

    return chosen_sorted, bonus_num, frames

# Sidebar controls
num_combos = st.sidebar.slider("Number of combinations to generate", min_value=1, max_value=10, value=1)

if st.button("Generate Lotto Numbers"):
    for idx in range(num_combos):
        combo, bonus, prob_frames = generate_lotto_numbers()
        exp = st.expander(f"Combination {idx+1}")
        with exp:
            st.subheader(f"Numbers: {combo}  •  Bonus: {bonus}")

            # Show ranges for each main number
            ranges = []
            for n in combo:
                start = max(1, n - 3)
                end = min(49, n + 3)
                ranges.append({"Number": n, "Range Start": start, "Range End": end})
            range_df = pd.DataFrame(ranges)

            st.write("**Number Ranges (±3)**")
            st.table(range_df)

            # Bubble/rule chart for main numbers
            rule = alt.Chart(range_df).mark_rule(color='gray', size=3).encode(
                x='Range Start:Q',
                x2='Range End:Q'
            )
            circles = alt.Chart(range_df).mark_circle(color='red', size=100).encode(
                x='Number:Q',
                tooltip=['Number', 'Range Start', 'Range End']
            )
            chart = alt.layer(rule, circles).properties(
                width=600,
                height=100,
                title='Main Number Ranges'
            )
            st.altair_chart(chart, use_container_width=True)

            # Probability bar charts
            for title, df in prob_frames:
                st.subheader(title)
                st.bar_chart(df.set_index(df.columns[0]))
