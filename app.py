import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(layout="centered")
st.title("🎯 Learn Python by Moving the Ball!")

# Initialize position
if "x" not in st.session_state:
    st.session_state.x = 5
    st.session_state.y = 5

# Movement functions
def move_right():
    st.session_state.x = min(st.session_state.x + 1, 9)

def move_left():
    st.session_state.x = max(st.session_state.x - 1, 0)

def move_up():
    st.session_state.y = max(st.session_state.y - 1, 0)

def move_down():
    st.session_state.y = min(st.session_state.y + 1, 9)

# Draw canvas
def render_canvas(x, y):
    canvas_size = 300
    grid_size = 10
    cell_size = canvas_size // grid_size

    img = Image.new("RGB", (canvas_size, canvas_size), color="white")
    draw = ImageDraw.Draw(img)

    for i in range(0, canvas_size, cell_size):
        draw.line((i, 0, i, canvas_size), fill="gray")
        draw.line((0, i, canvas_size, i), fill="gray")

    ball_pos = (x * cell_size, y * cell_size,
                (x + 1) * cell_size, (y + 1) * cell_size)
    draw.ellipse(ball_pos, fill="blue")

    st.image(img)

# Instruction and code input
st.write("### 💡 Try: `move_right()`, `move_up()`, `move_left()`, `move_down()`")
user_code = st.text_area("Enter your Python code below:", height=150)

render_canvas(st.session_state.x, st.session_state.y)

if st.button("Run Code"):
    try:
        exec(user_code, {"move_right": move_right,
                         "move_left": move_left,
                         "move_up": move_up,
                         "move_down": move_down})
        st.success("✅ Success! Ball moved.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
