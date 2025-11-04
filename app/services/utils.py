import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

DB_PROCESSED_PATH = '../../data/db/obesity_data_processed.db'
DB_SPLIT_PATH = '../../data/db/split_data.db'

def load_and_split_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load data from the processed DB, correct the target variable, and perform a stratified split.
    """
    try:
        conn = sqlite3.connect(DB_PROCESSED_PATH)
        processed_data = pd.read_sql_query("SELECT * FROM obesity_data_processed", conn)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

    features = processed_data.drop(["NObeyesdad", "timestamp"], axis=1)
    target = processed_data["NObeyesdad"]
    
    # FIX: Reverse the MinMax scaling on the target (0.0 to 1.0 -> 0 to 6)
    target = (target * 6).round().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    return X_train, X_test, y_train, y_test

def log_predictions(model_name: str, y_true, y_pred: np.ndarray, source: str = "train"):
    """Logs predictions to the 'predictions' table in the split_data.db."""
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    actual_values = y_true.values if isinstance(y_true, pd.Series) else y_true

    log_df = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()] * len(y_pred),
        "model": [model_name] * len(y_pred),
        "source": [source] * len(y_pred),
        "actual": actual_values,
        "predicted": y_pred
    })
    
    try:
        conn = sqlite3.connect(DB_SPLIT_PATH)
        log_df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.close()
        # In a real app, you might use an async DB library or a background task
    except sqlite3.Error as e:
        print(f"Error logging predictions: {e}")