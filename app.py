import streamlit as st
import random
import openai
import os

# Set your OpenAI API key as an environment variable or via Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Question generators for school-leaver math level

def generate_linear_question():
    a = random.randint(1, 10)
    b = random.randint(-20, 20)
    question = f"Solve for x: {a}x + {b} = 0"
    solution = -b / a
    return question, solution

# Add more generators here (e.g., quadratic, basic calculus)
def generate_random_question():
    generators = [generate_linear_question]
    return random.choice(generators)()

# Call OpenAI to get a hint

def get_hint(question, student_answer):
    messages = [
        {"role": "system", "content": "You are a patient math tutor. Provide a helpful hint that guides the student toward the solution without giving it away."},
        {"role": "user", "content": f"Question: {question}\nStudent's answer: {student_answer}\nGive a hint to help the student."}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content

# Initialize or reset session state
if "question" not in st.session_state or st.button("New Question"):
    q, sol = generate_random_question()
    st.session_state.question = q
    st.session_state.solution = sol
    st.session_state.history = []

st.title("📚 Math Tutor for School Leavers")
st.subheader("Solve the following question:")
st.write(f"**{st.session_state.question}**")

# Student answer input
user_input = st.text_input("Your answer:")
if st.button("Submit Answer"):
    if not user_input.strip():
        st.warning("Please enter an answer.")
    else:
        try:
            # Numeric comparison; adjust tolerance as needed
            student_val = float(user_input)
            if abs(student_val - st.session_state.solution) < 1e-3:
                st.success("Correct! 🎉 Well done.")
            else:
                hint = get_hint(st.session_state.question, user_input)
                st.session_state.history.append(hint)
        except ValueError:
            st.error("Please enter a valid numeric answer.")

# Display hints/chat history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Hints from your tutor:")
    for i, h in enumerate(st.session_state.history, 1):
        st.info(f"**Hint {i}:** {h}")

# Instructions to run
st.markdown("---")
st.write("Run this app with: `streamlit run streamlit_math_tutor.py` and make sure your OpenAI API key is set.")
