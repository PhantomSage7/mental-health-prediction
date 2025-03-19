# code\modeling\7_combined_model_training.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import DATA_PATHS, MODEL_PATHS
from model_utils import create_pipeline
import pandas as pd

def train_combined_model():
    df = pd.read_csv(DATA_PATHS['combined'])
    
    # Dynamically check which columns exist before dropping
    columns_to_drop = ['ema_score', 'study_id', 'date']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=existing_columns)
    y = df['ema_score'].apply(lambda x: 1 if x > np.median(df['ema_score']) else 0)
    
    pipeline = create_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    )
    pipeline.fit(X, y)
    
    # Ensure directory exists
    MODEL_PATHS['combined'].mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, MODEL_PATHS['combined']/'combined_model.pkl')
    print(f"Saved combined model to {MODEL_PATHS['combined']/'combined_model.pkl'}")

if __name__ == '__main__':
    train_combined_model()