import streamlit as st
import random

# Question generators for school-leaver math level

def generate_addition_word_problem():
    x = random.randint(5, 20)
    y = random.randint(1, 15)
    question = f"Sarah has {x} apples and buys {y} more. How many apples does she have now?"
    solution = x + y
    hints = [
        f"What operation combines {x} and {y}? Consider adding them.",
        f"Calculate {x} + {y} to find the total number of apples.",
    ]
    return question, solution, hints


def generate_subtraction_word_problem():
    x = random.randint(20, 50)
    y = random.randint(1, x - 1)
    question = f"James had {x} marbles but gave {y} to his friend. How many marbles does he have left?"
    solution = x - y
    hints = [
        f"Think about removing {y} from {x}. What operation do you use?",
        f"Compute {x} - {y} to get the remaining marbles.",
    ]
    return question, solution, hints


def generate_multiplication_word_problem():
    x = random.randint(2, 10)
    y = random.randint(2, 10)
    question = f"Each pack contains {x} pencils. If you buy {y} packs, how many pencils do you get in total?"
    solution = x * y
    hints = [
        f"Multiplication helps when you have {y} groups of {x}.",
        f"Calculate {x} * {y} to find the total pencils.",
    ]
    return question, solution, hints


def generate_division_word_problem():
    total = random.randint(20, 100)
    parts = random.randint(2, 10)
    question = f"{total} cookies are shared equally among {parts} children. How many cookies does each child get?"
    solution = total / parts
    hints = [
        f"Division splits {total} into {parts} equal parts.",
        f"Compute {total} ÷ {parts} to find how many per child.",
    ]
    return question, solution, hints


def generate_random_question():
    generators = [
        generate_addition_word_problem,
        generate_subtraction_word_problem,
        generate_multiplication_word_problem,
        generate_division_word_problem,
    ]
    return random.choice(generators)()

# Initialize or reset session state
if "question" not in st.session_state or st.button("New Question", key="new_question_init"):
    q, sol, hint_list = generate_random_question()
    st.session_state.question = q
    st.session_state.solution = sol
    st.session_state.hints = hint_list
    st.session_state.hint_index = 0
    st.session_state.completed = False
    st.session_state.user_answer = None

st.title("📚 Interactive Math Word Problems")
st.subheader("Try to solve this word problem:")
st.write(f"**{st.session_state.question}**")

# User input section
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.number_input("Your answer:", value=0.0, key="user_answer")
with col2:
    if st.button("Get Hint", key="get_hint") and not st.session_state.completed:
        if st.session_state.hint_index < len(st.session_state.hints):
            hint = st.session_state.hints[st.session_state.hint_index]
            st.info(f"Hint {st.session_state.hint_index + 1}: {hint}")
            st.session_state.hint_index += 1
        else:
            st.warning("No more hints available.")

if st.button("Check Answer", key="check_answer") and not st.session_state.completed:
    try:
        if abs(user_input - st.session_state.solution) < 1e-3:
            st.success("🎉 Correct! Great job.")
            st.session_state.completed = True
        else:
            st.error("That’s not quite right. Try again or get a hint!")
    except Exception:
        st.error("Please enter a valid number.")

# Completion message and next steps
if st.session_state.completed:
    st.balloons()
    st.write(f"The correct answer was **{st.session_state.solution}**.")
    if st.button("New Question", key="new_question_completed"):
        # Clear question to trigger new one
        del st.session_state["question"]

st.markdown("---")
st.write("Run with: `streamlit run streamlit_math_tutor.py`. No API key needed.")
