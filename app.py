import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="AI Model Playground")

st.title("🤖 AI Model Playground")
st.markdown("""
An interactive tool to learn how neural network hyperparameters affect
model performance on a simple dataset.
""")

# Generate synthetic dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Sidebar: hyperparameters
st.sidebar.header("Hyperparameters")
hidden_layers = st.sidebar.slider("Hidden Layer Sizes", 1, 5, (3, 3), help="Tuple of layer sizes")
activation = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"])
learning_rate = st.sidebar.select_slider("Learning Rate", options=[1e-1, 1e-2, 1e-3, 1e-4], value=1e-3)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize model
model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                      activation=activation,
                      learning_rate_init=learning_rate,
                      max_iter=1,
                      warm_start=True,
                      random_state=1)

# Training loop with live plotting
history = []
for epoch in range(1, epochs + 1):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    history.append(score)
    if epoch % (epochs // 5) == 0 or epoch == epochs:
        st.write(f"Epoch {epoch}/{epochs} - Test Accuracy: {score:.3f}")

# Layout: two columns for plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Decision Boundary")
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, edgecolor='k')
    ax.set_title("Test Set Decision Boundary")
    st.pyplot(fig)

with col2:
    st.subheader("Accuracy over Epochs")
    df_hist = pd.DataFrame({'Accuracy': history})
    st.line_chart(df_hist)

st.markdown("---")
st.write("Adjust the sliders to see how hyperparameters impact accuracy and decision boundary.")
