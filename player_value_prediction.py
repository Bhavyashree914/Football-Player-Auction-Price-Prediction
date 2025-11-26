import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("real_player_prices.csv")

# Encode categorical features
df['position'] = LabelEncoder().fit_transform(df['position'])
df['nationality'] = LabelEncoder().fit_transform(df['nationality'])

# Define features and target
X = df[['age', 'position', 'nationality']]
y = df['market_value_in_cr']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ₹{mae:.2f} Cr")
print(f"R² Score: {r2:.2f}")

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/player_value_model.pkl")
print("Model saved to model/player_value_model.pkl")
