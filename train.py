import joblib, os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.feature_extractor import extract_features

train_path = "MLDemoProj/Training_set/Training_set" 
test_path  = "MLDemoProj/test/test"

print("Extracting features...")
X_train, y_train = extract_features(train_path)
X_test, y_test   = extract_features(test_path)

# Check if data is empty
if len(X_train) == 0:
    print(f"ERROR: No images found in training path: {train_path}")
    print("Make sure you have:")
    print("  - electric_bus/ folder with images")
    print("  - electric_car/ folder with images")
    exit(1)
if len(X_test) == 0:
    print(f"ERROR: No images found in test path: {test_path}")
    exit(1)

print(f"✓ Loaded {len(X_train)} training samples")
print(f"✓ Loaded {len(X_test)} test samples")

models = {
    "SVM": SVC(kernel="linear", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_model = None
best_accuracy = 0

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc}")

    joblib.dump(model, f"models/{name}.pkl")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Model Saved!")
