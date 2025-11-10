import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Starting IPL model training...")

# Load the dataset
matches = pd.read_csv("all_matches.csv")

# Basic cleaning
matches = matches.dropna(subset=["team1", "team2", "venue", "toss_decision", "winner"])
matches = matches[matches["team1"] != matches["team2"]]

# Initialize encoders
team_encoder = OrdinalEncoder()
venue_encoder = OrdinalEncoder()
toss_decision_encoder = OrdinalEncoder()

# Encode categorical columns
matches["team1_enc"] = team_encoder.fit_transform(matches[["team1"]])
matches["team2_enc"] = team_encoder.transform(matches[["team2"]])
matches["venue_enc"] = venue_encoder.fit_transform(matches[["venue"]])
matches["toss_decision_enc"] = toss_decision_encoder.fit_transform(matches[["toss_decision"]])
matches["target"] = team_encoder.transform(matches[["winner"]])

# Features and target
X = matches[["team1_enc", "team2_enc", "venue_enc", "toss_decision_enc"]]
y = matches["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully with accuracy: {accuracy:.3f}")

# Save model and encoders
pickle.dump(model, open("ipl_winner_model.pkl", "wb"))
pickle.dump(team_encoder, open("team_encoder.pkl", "wb"))
pickle.dump(venue_encoder, open("venue_encoder.pkl", "wb"))
pickle.dump(toss_decision_encoder, open("toss_decision_encoder.pkl", "wb"))

print("All pickle files have been saved successfully!")
