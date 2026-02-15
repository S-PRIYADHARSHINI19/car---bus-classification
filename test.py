import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from utils.feature_extractor import extract_features

X_test, y_test = extract_features("MLDemoProj/test/test")

# Check if data is empty
if len(X_test) == 0:
    print("❌ ERROR: No test images found!")
    print("Please add images to:")
    print("  - MLDemoProj/test/test/electric bus/")
    print("  - MLDemoProj/test/test/electric car/")
    exit(1)

if not os.path.exists("models/best_model.pkl"):
    print("❌ ERROR: Trained model not found!")
    print("Run 'python train.py' first to train the model.")
    exit(1)

model = joblib.load("models/best_model.pkl")
pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
