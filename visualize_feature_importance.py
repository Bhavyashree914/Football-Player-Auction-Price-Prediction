import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model/player_value_model.pkl")
df = pd.read_csv("dataset.csv")

df['player_value'] = (
    df['events.goals'] * 10 +
    df['events.assists'] * 7 +
    df['minutes_played'] * 0.1 +
    df['events.clean_sheet'] * 5 +
    df['events.yellow_cards'] * -2 +
    df['events.red_cards'] * -5
)

df = df.select_dtypes(include=['number']).dropna()
X = df.drop(columns=['player_value'])

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
