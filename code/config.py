#code\config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    'crosscheck_raw': BASE_DIR/'data/raw/crosscheck/crosscheck_daily_data_cleaned_w_sameday.csv',
    'studentlife_raw': BASE_DIR/'data/raw/studentlife/studentlife_daily_data_cleaned_w_sameday_08282021.csv',
    'crosscheck_processed': BASE_DIR/'data/processed/crosscheck_preprocessed.csv',
    'studentlife_processed': BASE_DIR/'data/processed/studentlife_preprocessed.csv',
    'combined': BASE_DIR/'data/combined/combined_dataset.csv',
    'combined_processed': BASE_DIR/'data/combined/combined_dataset.csv'
}

MODEL_PATHS = {
    'crosscheck': BASE_DIR/'models/crosscheck_models/',
    'studentlife': BASE_DIR/'models/studentlife_models/',
    'combined': BASE_DIR/'models/combined_models/'
}

FEATURE_SETTINGS = {
    'target': 'ema_score',  # Common target variable
    'drop_cols': ['study_id', 'date', 'eureka_id']
}

for path in MODEL_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

(DATA_PATHS['combined'].parent).mkdir(parents=True, exist_ok=True)
