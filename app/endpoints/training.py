import os
from fastapi import APIRouter, HTTPException
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy.stats import randint as sp_randint, loguniform

from ..services.utils import load_and_split_data, log_predictions
# from ..database import crud # Add DB interaction for logging metrics
# from ..schemas import TrainingStatus # Define a schema for response

router = APIRouter()

@router.post("/train-and-optimize", summary="Retrain and optimize all ML models")
def run_full_training_pipeline():
    X_train, X_test, y_train, y_test = load_and_split_data()
    if X_train is None:
        raise HTTPException(status_code=500, detail="Could not load training data.")

    # --- Setup ---
    os.makedirs("models", exist_ok=True)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metric = 'f1_weighted'
    optimized_models = {}
    
    # --- Optimization Logic (Simplified, repeat for RF, CatBoost, SVC) ---
    
    # 1. CatBoost Optimization (Example)
    cb_param_dist = {
        'learning_rate': [0.05, 0.1, 0.2], 'depth': sp_randint(4, 8),
        'iterations': [200, 400]
    }
    cb_random_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, loss_function='MultiClass'),
        param_distributions=cb_param_dist, n_iter=10, cv=cv_strategy, scoring=scoring_metric,
        random_state=42, n_jobs=-1
    )
    cb_random_search.fit(X_train, y_train)
    best_cb_model = cb_random_search.best_estimator_
    optimized_models["CatBoostClassifier"] = best_cb_model

    # ... Repeat RandomizedSearchCV for RF and SVC ...

    # --- Final Evaluation and Saving ---
    best_f1 = -1
    best_model_name = ""
    
    for name, model in optimized_models.items():
        y_pred_test = model.predict(X_test)
        log_predictions(f"{name}_Optimized", y_test, y_pred_test, source="test")
        
        test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

        if test_f1_weighted > best_f1:
            best_f1 = test_f1_weighted
            best_model_name = name
            joblib.dump(model, f'models/final_model_{name.lower()}.pkl') # Save specific model
            
    # Save the final best model
    final_model = optimized_models[best_model_name]
    joblib.dump(final_model, 'models/final_model_best.pkl') # Save a generic best version
    
    return {"status": "success", "message": f"Training complete. Best model: {best_model_name} (F1: {best_f1:.4f})"}