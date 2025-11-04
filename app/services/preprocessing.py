import pandas as pd
import numpy as np
from typing import Dict, Any

# --- Масштабування та Кодування (Зібрано з Lab 2) ---
# 1. Словники для Label Encoding (OHE-подібне кодування, яке використовує LabelEncoder)
# Ці значення були отримані в результаті fit_transform на етапі Lab 2
ENCODERS = {
    'Gender': {'Female': 0, 'Male': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'SMOKE': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'CAEC': {'Always': 0, 'Frequently': 1, 'Never': 2, 'Sometimes': 3},
    'CALC': {'Always': 0, 'Frequently': 1, 'Never': 2, 'Sometimes': 3},
    # MTRANS - це One-Hot Encoding в цілому, але якщо використовувався LabelEncoder, 
    # маємо використовувати це ж відображення.
    'MTRANS': {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4} 
}

# 2. Параметри MinMax Scaler
MIN_MAX_PARAMS = {
    'Age': {'min': 14.0, 'max': 61.0},
    'Height': {'min': 1.45, 'max': 1.98},
    'Weight': {'min': 39.0, 'max': 173.0},
    'FCVC': {'min': 1.0, 'max': 3.0},
    'NCP': {'min': 1.0, 'max': 4.0},
    'CH2O': {'min': 1.0, 'max': 3.0},
    'FAF': {'min': 0.0, 'max': 3.0},
    'TUE': {'min': 0.0, 'max': 2.0},
}

def scale_value(value: float, key: str) -> float:
    """Applies MinMax scaling (X - min) / (max - min)"""
    params = MIN_MAX_PARAMS.get(key)
    if params:
        min_val = params['min']
        max_val = params['max']
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)
    return value

def preprocess_raw_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms raw input data from the user into the scaled/encoded DataFrame 
    format expected by the ML model.
    """
    processed_data = {}
    
    # --- 1. Обробка бінарних та номінальних категоріальних фіч (Label Encoding) ---
    for key, mapping in ENCODERS.items():
        raw_val = raw_data.pop(key)
        # У Lab 2 ми використовували LabelEncoder. Тут ми імітуємо його поведінку
        # на основі фіксованих порядків, які LE присвоїв.
        processed_data[key] = mapping.get(raw_val, -1) # -1 is a fallback
        if processed_data[key] == -1:
            raise ValueError(f"Invalid value for {key}: {raw_val}")

    # --- 2. Обробка числових фіч (MinMax Scaling) ---
    for key, params in MIN_MAX_PARAMS.items():
        raw_val = raw_data.pop(key)
        processed_data[key] = scale_value(raw_val, key)

    # --- 3. Обробка ORDINAL/Continuous features (які не були в ENCODERS/MIN_MAX_PARAMS) ---
    FEATURE_ORDER = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
        'CALC', 'MTRANS'
    ]
    
    final_input = pd.DataFrame([[processed_data[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return final_input

MTRANS_MAPPING = {
    'Automobile': 0.0, 'Bike': 1.0, 'Motorbike': 2.0, 
    'Public_Transportation': 3.0, 'Walking': 4.0
}

MTRANS_ENCODING = {
    'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4
}

ENCODERS['MTRANS'] = MTRANS_ENCODING
