import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import joblib

# =========================
# Load dataset
# =========================
df = pd.read_csv("apple_quality.csv")

print("Columns in CSV:", df.columns)
print(df.head())

# Convert all feature columns to numeric (handle object issue)
feature_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN (if any)
df = df.dropna()

# Features (X) and Labels (y)
X = df[feature_cols].values.astype('float32')
y = df["Quality"].values

# Encode labels (good=0, bad=1, average=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =========================
# Define Neural Network
# =========================
class AppleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AppleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = len(feature_cols)   # 7 features
hidden_size = 16
num_classes = len(label_encoder.classes_)  # good, bad, average

model = AppleNet(input_size, hidden_size, num_classes)

# =========================
# Loss and Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training Loop
# =========================
epochs = 50
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =========================
# Save Model and Label Encoder
# =========================
torch.save(model.state_dict(), "apple_model.pth")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Training complete. Model and label encoder saved.")
