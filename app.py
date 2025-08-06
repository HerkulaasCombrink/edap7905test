import streamlit as st
import random

# Question generators for school-leaver math level
def generate_linear_question():
    a = random.randint(1, 10)
    b = random.randint(-20, 20)
    question = f"Solve for x: {a}x + {b} = 0"
    solution = -b / a
    hints = [
        f"First, subtract {b} from both sides to isolate the term with x.",
        f"Now you have {a}x = {-b}. Divide both sides by {a} to solve for x.",
    ]
    return question, solution, hints

# Add more generators here (e.g., quadratic) and their hints

def generate_random_question():
    generators = [generate_linear_question]
    return random.choice(generators)()

# Initialize or reset session state
if "question" not in st.session_state or st.button("New Question"):
    q, sol, hint_list = generate_random_question()
    st.session_state.question = q
    st.session_state.solution = sol
    st.session_state.hints = hint_list
    st.session_state.hint_index = 0
    st.session_state.completed = False

st.title("📚 Math Tutor for School Leavers")
st.subheader("Solve the following question:")
st.write(f"**{st.session_state.question}**")

# Student answer input
user_input = st.text_input("Your answer:")
if st.button("Submit Answer") and not st.session_state.completed:
    if not user_input.strip():
        st.warning("Please enter an answer.")
    else:
        try:
            student_val = float(user_input)
            if abs(student_val - st.session_state.solution) < 1e-3:
                st.success("Correct! 🎉 Well done.")
                st.session_state.completed = True
            else:
                idx = st.session_state.hint_index
                if idx < len(st.session_state.hints):
                    hint = st.session_state.hints[idx]
                    st.info(f"Hint {idx+1}: {hint}")
                    st.session_state.hint_index += 1
                else:
                    st.error(f"No more hints available. The solution is x = {st.session_state.solution}")
                    st.session_state.completed = True
        except ValueError:
            st.error("Please enter a valid numeric answer.")

# Instructions to run
st.markdown("---")
st.write("Run this app with: `streamlit run streamlit_math_tutor.py`. No API key is needed—hints are generated locally.")
