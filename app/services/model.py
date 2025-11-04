import joblib
import os
from sklearn.base import ClassifierMixin

MODEL_PATH = 'models/final_model_catboost.pkl' # Assuming CatBoost is the best

def load_best_model() -> ClassifierMixin:
    """Load the best trained model from the models directory."""
    full_path = os.path.join(os.getcwd(), MODEL_PATH)
    
    if not os.path.exists(full_path):
        # Fallback in a real scenario
        raise FileNotFoundError(f"Best model not found at {full_path}")
        
    return joblib.load(MODEL_PATH)