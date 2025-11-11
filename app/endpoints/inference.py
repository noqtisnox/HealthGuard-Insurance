from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
from typing import Dict, Any

from ..services.preprocessing import preprocess_raw_data

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_SPLIT_PATH = os.path.join(BASE_DIR, '../../data/db/split_data.db')
DB_INFERENCE_LOG_PATH = os.path.join(BASE_DIR, '../../data/db/inference_log.db')
DB_PROCESSED_PATH = os.path.join(BASE_DIR, '../../data/db/obesity_data_processed.db') # Додано для повноти

# Виносимо створення таблиць для чистоти
def create_tables_if_not_exist():
    """Create the necessary database tables if they don't exist"""
    
    # Створення таблиці predictions у split_data.db
    try:
        conn = sqlite3.connect(DB_SPLIT_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp TEXT,
                model TEXT,
                source TEXT,
                actual REAL,
                predicted REAL
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error creating predictions table in {DB_SPLIT_PATH}: {e}")

    # Створення таблиці inference_inputs у inference_log.db
    try:
        conn = sqlite3.connect(DB_INFERENCE_LOG_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_inputs (
                timestamp TEXT,
                prediction_coded INTEGER,
                Gender TEXT,
                Age REAL,
                Height REAL,
                Weight REAL,
                family_history_with_overweight TEXT,
                FAVC TEXT,
                FCVC REAL,
                NCP REAL,
                CAEC TEXT,
                SMOKE TEXT,
                CH2O REAL,
                SCC TEXT,
                FAF REAL,
                TUE REAL,
                CALC TEXT,
                MTRANS TEXT
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error creating inference_inputs table in {DB_INFERENCE_LOG_PATH}: {e}")

# Викликаємо створення таблиць
create_tables_if_not_exist()


def log_predictions(model_name: str, y_true, y_pred: np.ndarray, source: str = "train"):
    """
    Logs model predictions (actual vs predicted) to the 'predictions' table in split_data.db.
    Used for logging training/test evaluation and inference results.
    """
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    # Обробка y_true для інференсу: якщо передано [None], то використовуємо np.nan
    actual_values = y_true.values if isinstance(y_true, pd.Series) else y_true
    if actual_values is not None and len(actual_values) > 0 and actual_values[0] is None:
        actual_values = np.array([np.nan] * len(y_pred))

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
    except sqlite3.Error as e:
        print(f"Error logging predictions for {model_name} ({source}): {e}")

def log_inference_input(raw_data: Dict[str, Any], prediction_coded: int):
    """
    Logs raw input data and the final prediction result to a dedicated table (inference_inputs).
    """
    timestamp = datetime.now().isoformat()
    
    # Build the data dictionary in the correct order
    data_to_log = {
        'timestamp': timestamp,
        'prediction_coded': int(prediction_coded),
        'Gender': raw_data.get('Gender'),
        'Age': raw_data.get('Age'),
        'Height': raw_data.get('Height'),
        'Weight': raw_data.get('Weight'),
        'family_history_with_overweight': raw_data.get('family_history_with_overweight'),
        'FAVC': raw_data.get('FAVC'),
        'FCVC': raw_data.get('FCVC'),
        'NCP': raw_data.get('NCP'),
        'CAEC': raw_data.get('CAEC'),
        'SMOKE': raw_data.get('SMOKE'),
        'CH2O': raw_data.get('CH2O'),
        'SCC': raw_data.get('SCC'),
        'FAF': raw_data.get('FAF'),
        'TUE': raw_data.get('TUE'),
        'CALC': raw_data.get('CALC'),
        'MTRANS': raw_data.get('MTRANS')
    }
    
    # Створюємо DataFrame - the dict already has the right keys, so just create from it
    input_df = pd.DataFrame([data_to_log])
    
    try:
        conn = sqlite3.connect(DB_INFERENCE_LOG_PATH)
        input_df.to_sql("inference_inputs", conn, if_exists="append", index=False)
        conn.close()
        print("Successfully logged inference input to DB")
        return True
    except sqlite3.Error as e:
        print(f"Error logging inference input to DB: {e}")
        print(f"Data that failed to insert: {data_to_log}") 
        return False


templates = Jinja2Templates(directory="app/templates") 
router = APIRouter()

try:
    LOADED_MODEL = joblib.load('models/final_model_catboost.pkl')
except FileNotFoundError:
    LOADED_MODEL = None
    print("Warning: Model not found. Prediction endpoint will fail.")


# --- 1. Ендпоінт для відображення форми ---
@router.get("/predict", response_class=HTMLResponse, summary="Show web form for prediction")
async def get_prediction_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- 2. Ендпоінт для обробки даних форми ---
@router.post("/predict-web", response_class=HTMLResponse, summary="Get prediction from web form")
async def submit_prediction_form(
    request: Request,
    # Числові (сирі)
    Gender_raw: str = Form(..., alias="Gender"),
    Age_raw: float = Form(..., alias="Age"),
    Height_raw: float = Form(..., alias="Height"),
    Weight_raw: float = Form(..., alias="Weight"),
    FCVC_raw: float = Form(..., alias="FCVC"), # Frequency of vegetable consumption (1-3)
    NCP_raw: float = Form(..., alias="NCP"),   # Number of main meals per day (1-4)
    CH2O_raw: float = Form(..., alias="CH2O"), # Daily water intake (1-3)
    FAF_raw: float = Form(..., alias="FAF"),   # Physical activity frequency (0-3)
    TUE_raw: float = Form(..., alias="TUE"),   # Time spent using technology (0-2)
    
    family_history_with_overweight_raw: str = Form(..., alias="family_history_with_overweight"),
    FAVC_raw: str = Form(..., alias="FAVC"),
    CAEC_raw: str = Form(..., alias="CAEC"),
    SMOKE_raw: str = Form(..., alias="SMOKE"),
    SCC_raw: str = Form(..., alias="SCC"),
    CALC_raw: str = Form(..., alias="CALC"),
    MTRANS_raw: str = Form(..., alias="MTRANS"),
):
    if LOADED_MODEL is None:
        return templates.TemplateResponse("index.html", {"request": request, "category": "Помилка: Модель не завантажена."})
    
    raw_input_data: Dict[str, Any] = {
        'Gender': Gender_raw,
        'Age': Age_raw,
        'Height': Height_raw,
        'Weight': Weight_raw,
        'family_history_with_overweight': family_history_with_overweight_raw,
        'FAVC': FAVC_raw,
        'FCVC': FCVC_raw,
        'NCP': NCP_raw,
        'CAEC': CAEC_raw,
        'SMOKE': SMOKE_raw,
        'CH2O': CH2O_raw,
        'SCC': SCC_raw,
        'FAF': FAF_raw,
        'TUE': TUE_raw,
        'CALC': CALC_raw,
        'MTRANS': MTRANS_raw,
    }
    
    # IMPORTANT: Save a copy BEFORE preprocessing, as preprocessing might modify the dict
    raw_input_data_backup = raw_input_data.copy()
    
    try:
        # --- КРОК 1: ПОПЕРЕДНЯ ОБРОБКА ---
        input_df = preprocess_raw_data(raw_input_data)
    except ValueError as e:
        return templates.TemplateResponse("index.html", {"request": request, "category": f"Помилка обробки даних: {e}"})

    # --- КРОК 2: ПРОГНОЗУВАННЯ ---
    prediction_coded = LOADED_MODEL.predict(input_df)[0]
    if hasattr(prediction_coded, 'item'):
        prediction_coded = prediction_coded.item()
    else:
        prediction_coded = float(prediction_coded)
    
    # Зворотне відображення прогнозованого коду (0-6) у категорію
    categories = [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", 
        "Obesity_Type_III"
    ]
    predicted_category = categories[int(prediction_coded)]


    log_inference_input(raw_input_data_backup, int(prediction_coded))    
    log_predictions("CatBoost", [None]*len(input_df), np.array([prediction_coded]), source="inference")
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "prediction": prediction_coded,
            "category": predicted_category,
            "input_data": raw_input_data
        }
    )
