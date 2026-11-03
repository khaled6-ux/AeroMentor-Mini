
import joblib
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Locate and load the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "flight_training_data.csv"

data = pd.read_csv(DATA_FILE)

# 2. Select our features (X)
features = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]

X = data[features]

# 3. Select our labels (y)
y = data["label"]

# 4. Split the flights into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Create a Decision Tree
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
)

# 6. Train the model
model.fit(X_train, y_train)

# Save the trained model
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "flight_classifier.joblib"

joblib.dump(model, MODEL_FILE)

print(f"Model saved to: {MODEL_FILE}")


# 7. Predict labels for unseen flights
predictions = model.predict(X_test)

# 8. Evaluate results
accuracy = accuracy_score(y_test, predictions)

print("AEROMENTOR MINI - ML CLASSIFICATION")
print("----------------------------------")
print(f"Training flights: {len(X_train)}")
print(f"Testing flights: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["stable", "unstable"],
    )
)



# Predict a new example flight
new_flight = pd.DataFrame(
    [{
        "altitude_mae_ft": 35,
        "airspeed_mae_kt": 2.5,
        "heading_mae_deg": 4
    }],
    columns=features
)

prediction = model.predict(new_flight)

print("\nNEW FLIGHT PREDICTION")
print("---------------------")
print(f"Predicted classification: {prediction[0]}")


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualize the trained Decision Tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("AeroMentor Mini - Decision Tree")

plt.tight_layout()
plt.show()
