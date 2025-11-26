import pandas as pd
import pickle

# Load trained model
with open("auction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load new dataset (players to predict)
new_data = pd.read_csv("new_players.csv")

# Keep original names/positions for display
player_info = new_data[["Name", "Position"]]

# Drop non-numeric columns
X_new = new_data.drop(columns=["Name", "Position"])

# Predict values
predictions = model.predict(X_new)

# Combine results
results = player_info.copy()
results["Predicted_AuctionPrice"] = predictions

# Save to CSV
results.to_csv("predicted_values.csv", index=False)

print("✅ Predictions saved to predicted_values.csv")
print(results)
