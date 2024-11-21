import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the features and labels
features_df = pd.read_csv("data/features.csv")

# Assuming you have build spot data (in a 'build_spots.csv' file) to use as labels
build_spots_df = pd.read_csv("data/build_spots.csv")

# Merge build spots with features (this assumes grid_id matches)
data = pd.merge(features_df, build_spots_df, on="grid_id", how="left")

# Define X (features) and y (labels)
X = data[["avg_red", "avg_green", "avg_blue"]]  # Use other features as needed
y = data["built"]  # 1 for built, 0 for not built (use your own labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, "models/rust_build_predictor.pkl")
print("Model training complete and saved!")
