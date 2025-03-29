import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import streamlit as st

# --- Config ---
IMAGE_DIR = "images"
LABELS_FILE = "labels.csv"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
MODEL_SAVE_PATH = "hand_signal_cnn.pth"
PICKLE_SAVE_PATH = "hand_signal_cnn.pkl"

st.title("🧠 Train CNN on Hand Signal Dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load labels ---
df = pd.read_csv(LABELS_FILE)
classes = sorted(df["label"].unique().tolist())
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
df["label_idx"] = df["label"].map(class_to_idx)

# --- Train/val split ---
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# --- Custom Dataset ---
class HandSignalDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        label = row["label_idx"]

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Datasets & Loaders ---
train_dataset = HandSignalDataset(train_df, IMAGE_DIR, transform)
val_dataset = HandSignalDataset(val_df, IMAGE_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Train button ---
if st.button("🚀 Start Training"):
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix({
                "batch": batch_idx + 1,
                "loss": f"{loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        st.write(f"🟢 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        st.write(f"✅ Validation Accuracy: {acc:.2%}")

    # Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    st.success("✅ Model weights saved as .pth")

    # Save Pickle
    # Save state_dict and prepare download
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Let user download .pth as binary
    with open(MODEL_SAVE_PATH, "rb") as f:
        st.download_button("📥 Download Trained Weights (.pth)", f, file_name="hand_signal_cnn.pth")
