import joblib
import numpy as np

def load_model(model_path="random_forest_model.pkl"):
    """Load the trained model from disk."""
    return joblib.load(model_path)

def predict_price(model, features):
    """Predict house price using the model and input features."""
    features_array = np.array(features).reshape(1, -1)
    return model.predict(features_array)[0]
