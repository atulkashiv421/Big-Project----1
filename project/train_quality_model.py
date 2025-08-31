import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# =======================
# STEP 1: Dataset load
# =======================
df = pd.read_csv("apple_quality.csv", on_bad_lines="skip")

# सिर्फ numeric वाली rows रखना (अगर कहीं string आ गई तो remove हो जाएगी)
df = df[pd.to_numeric(df["Size"], errors="coerce").notnull()]

print("✅ Cleaned dataset columns:", df.columns)
print("✅ Dataset shape:", df.shape)

# =======================
# STEP 2: Features & Target
# =======================
X = df.drop(columns=["A_id", "Quality"])   # Features (input)
y = df["Quality"]                          # Target (output: good/bad)

# =======================
# STEP 3: Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# STEP 4: Model Training
# =======================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =======================
# STEP 5: Accuracy Check
# =======================
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

# =======================
# STEP 6: Save Model
# =======================
with open("quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 Model saved as quality_model.pkl")
