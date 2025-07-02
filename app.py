import streamlit as st
from utils.movement import move_ball, get_position, render_canvas

st.set_page_config(layout="centered")
st.title("🎯 Learn Python by Moving the Ball!")

# Initialize position
if "x" not in st.session_state:
    st.session_state.x = 5
    st.session_state.y = 5

st.write("### 🧪 Try writing Python code below to move the ball!")
st.code("Try: move_right(), move_up(), move_left(), move_down()")

user_code = st.text_area("Enter your Python code below", height=150)

# Show current canvas
render_canvas(st.session_state.x, st.session_state.y)

if st.button("Run Code"):
    try:
        local_scope = {}
        exec(user_code, {"move_right": lambda: move_ball("right"),
                         "move_left": lambda: move_ball("left"),
                         "move_up": lambda: move_ball("up"),
                         "move_down": lambda: move_ball("down")},
             local_scope)
        st.success("✅ Code executed successfully! Ball moved.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
