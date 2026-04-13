import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (replace later with real dataset if you want)
data = pd.DataFrame({
    "N": [50, 30, 60, 20, 70, 80, 25, 55],
    "pH": [6.5, 5.5, 7.0, 8.0, 6.8, 7.2, 5.8, 6.9],
    "TDS": [400, 1200, 300, 1500, 600, 450, 1300, 500],
    "Label": [1, 0, 1, 0, 1, 1, 0, 1]  # 1 = Good, 0 = Bad
})

X = data[["N", "pH", "TDS"]]
y = data["Label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("soil_water_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as soil_water_model.pkl")