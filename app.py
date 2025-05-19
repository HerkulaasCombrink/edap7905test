import streamlit as st
import requests
import base64
import os

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # Store safely
REPO_NAME = "HerkulaasCombrink/edap7905test"
FILE_PATH = "data/new_file.txt"
COMMIT_MESSAGE = "Add new_file.txt"
BRANCH = "main"

def push_to_github(content):
    api_url = f"https://api.github.com/repos/{REPO_NAME}/contents/{FILE_PATH}"
    encoded_content = base64.b64encode(content.encode()).decode()

    payload = {
        "message": COMMIT_MESSAGE,
        "content": encoded_content,
        "branch": BRANCH
    }

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.put(api_url, json=payload, headers=headers)
    return response.json()

# --- Streamlit UI
st.title("GitHub Push Demo")
user_input = st.text_area("Enter content to push to GitHub:")

if st.button("Push to GitHub"):
    result = push_to_github(user_input)
    if "content" in result:
        st.success("File pushed successfully!")
    else:
        st.error(f"Error: {result}")
