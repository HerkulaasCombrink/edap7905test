import streamlit as st
import pandas as pd

def generate_annotation(h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2):
    math_annotation = (
        f"({h},{c},{n}) \\times \\left[ "
        f"\\frac{{{d2}}}{{{d1}}} \\Bigg| "
        f"\\frac{{{l2}}}{{{l1}}} \\Bigg| "
        f"\\frac{{{s2}}}{{{s1}}} \\Bigg| "
        f"\\frac{{{p2}}}{{{p1}}} \\Bigg| "
        f"\\frac{{{m2}}}{{{m1}}} \\Bigg| "
        f"\\frac{{{r2}}}{{{r1}}} "
        f"\\right]"
    )

    csv_row = [h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2]
    return math_annotation, csv_row

st.title("SASL Annotation Generator")

# Layout for global parameters
st.subheader("Global Parameters")
col_global1, col_global2, col_global3 = st.columns(3)
with col_global1:
    h = st.selectbox("Handedness (H)", [1, 2], index=0)
with col_global2:
    c = st.selectbox("Contact (C)", [0, 1], index=0)
with col_global3:
    n = st.selectbox("Mouthing (N)", [1, 2], index=0)

# Layout for dominant and non-dominant hand
st.subheader("Dominant and Non-Dominant Hand Parameters")
col1, col2 = st.columns(2)
with col1:
    st.text("Dominant Hand")
    d1 = 1  # Always dominant hand
    l1 = st.slider("Location (L1)", 0, 10, 1)
    s1 = st.text_input("Handshape (S1)", "B")
    p1 = st.slider("Palm Orientation (P1)", 1, 5, 1)
    m1 = st.slider("Movement (M1)", 0, 23, 4)
    r1 = st.slider("Repetition (R1)", 0, 1, 1)

with col2:
    st.text("Non-Dominant Hand")
    d2 = st.selectbox("Non-Dominant Hand (D2)", [0, 2], index=0)
    l2 = st.slider("Location (L2)", 0, 10, 0)
    s2 = st.text_input("Handshape (S2)", "0")
    p2 = st.slider("Palm Orientation (P2)", 0, 5, 0)
    m2 = st.slider("Movement (M2)", 0, 23, 0)
    r2 = st.slider("Repetition (R2)", 0, 1, 0)

# Calculate button
if st.button("Calculate Annotation"):
    math_annotation, csv_row = generate_annotation(h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2)

    # Display results
    st.subheader("Mathematical Notation")
    st.latex(math_annotation)

    st.subheader("CSV Output")
    st.write(",".join(map(str, csv_row)))

    # Allow CSV Download
    df = pd.DataFrame([csv_row], columns=["H", "C", "N", "D1", "L1", "S1", "P1", "M1", "R1", "D2", "L2", "S2", "P2", "M2", "R2"])
    st.download_button("Download CSV", df.to_csv(index=False), "annotation.csv", "text/csv")
