import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(layout="centered")
st.title("🎯 Learn Python by Moving the Ball!")

# Target coordinates for the challenge
TARGET_X = 9
TARGET_Y = 0

# Initialize ball position
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

# Draw the canvas
def render_canvas(x, y, target=False):
    canvas_size = 300
    grid_size = 10
    cell_size = canvas_size // grid_size

    img = Image.new("RGB", (canvas_size, canvas_size), color="white")
    draw = ImageDraw.Draw(img)

    for i in range(0, canvas_size, cell_size):
        draw.line((i, 0, i, canvas_size), fill="gray")
        draw.line((0, i, canvas_size, i), fill="gray")

    # Draw target position
    if target:
        tx, ty = TARGET_X * cell_size, TARGET_Y * cell_size
        draw.rectangle([tx, ty, tx + cell_size, ty + cell_size], outline="red", width=3)

    # Draw ball
    ball_pos = (x * cell_size, y * cell_size,
                (x + 1) * cell_size, (y + 1) * cell_size)
    draw.ellipse(ball_pos, fill="blue")

    st.image(img)

# Challenge description
st.subheader("🧩 Challenge: Move the ball to the top-right corner")
st.markdown("Write code like `move_right()` and `move_up()` below to move the blue ball to the red box at **(x=9, y=0)**.")

# Code input
user_code = st.text_area("Enter your Python code below:", height=150)

# Show canvas
render_canvas(st.session_state.x, st.session_state.y, target=True)

# Check result
if st.button("Run Code"):
    try:
        exec(user_code, {
            "move_right": move_right,
            "move_left": move_left,
            "move_up": move_up,
            "move_down": move_down
        })
        if st.session_state.x == TARGET_X and st.session_state.y == TARGET_Y:
            st.success("🎉 Well done! You reached the target.")
        else:
            st.info("✅ Code ran! But the ball is not at the target yet. Try again.")
    except Exception as e:
        st.error(f"❌ Error in your code: {e}")
