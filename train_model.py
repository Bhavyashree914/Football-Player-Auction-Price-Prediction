import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

# --- Configuration ---
CSV_FILE = "players.csv"  # Your full CSV with AuctionPrice
MODEL_FILE = "auction_model.pkl"

# --- Load dataset ---
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Please place the full dataset in this folder.")

data = pd.read_csv(CSV_FILE)

# --- Check required columns ---
required_columns = ["Age", "Goals_2022", "Assists_2022", "Matches_2022",
                    "Goals_2023", "Assists_2023", "Matches_2023", "AuctionPrice"]

missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in CSV: {missing_cols}")

# --- Features and target ---
X = data.drop(columns=["AuctionPrice", "Name", "Position"], errors='ignore')
y = data["AuctionPrice"]

# --- Train model ---
model = GradientBoostingRegressor()
model.fit(X, y)

# --- Save model ---
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained and saved as {MODEL_FILE}")
