#code\modeling\6_individual_model_training.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from config import DATA_PATHS, MODEL_PATHS
from model_utils import create_pipeline


def train_individual_models():
    for dataset in ['crosscheck', 'studentlife']:
        # Load data
        df = pd.read_csv(DATA_PATHS[f'{dataset}_processed'])
        
        # Prepare features and target
        columns_to_drop = ['ema_score', 'study_id', 'date']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        X = df.drop(columns=existing_columns)
        
        # Convert target to binary classification
        y = df['ema_score'].apply(lambda x: 1 if x > np.median(df['ema_score']) else 0)
        
        # Create pipeline with class balancing
        pipeline = create_pipeline(
            GradientBoostingClassifier(
                random_state=42,
                subsample=0.8,
                min_samples_leaf=5  # Added regularization
            )
        )
        
        # Define hyperparameters
        param_grid = {
            'classifier__n_estimators': [100],
            'classifier__learning_rate': [0.1],
            'classifier__max_depth': [3]
        }
        
        # Train model with reduced CV folds
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=2,  # Reduced from 3 to 2
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=2
        )

        grid.fit(X, y)
        
        # Ensure model directory exists
        model_dir = MODEL_PATHS[dataset]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with protocol=4 for compatibility
        joblib.dump(grid.best_estimator_, model_dir / 'best_model.pkl', protocol=4)
        print(f"Saved {dataset} model to {model_dir}/best_model.pkl")

if __name__ == '__main__':
    train_individual_models()
