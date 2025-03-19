#code\feature_analysis\5_feature_selection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_PATHS, FEATURE_SETTINGS
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def select_features(dataset_path):
    df = pd.read_csv(dataset_path)
    
    # Convert potential string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                df = df.drop(columns=[col])

    # Ensure target is integer type
    try:
        df[FEATURE_SETTINGS['target']] = df[FEATURE_SETTINGS['target']].astype(int)
    except ValueError as e:
        print(f"Error converting target to integer: {e}")
        return []
    
    # Check target distribution
    target_counts = df[FEATURE_SETTINGS['target']].value_counts()
    print(f"Target distribution:\n{target_counts}")
    
    # Skip datasets with <2 classes
    if len(target_counts) < 2:
        print(f"Skipping {dataset_path} - insufficient classes")
        return []
    
    # Get existing columns to drop
    cols_to_drop = [col for col in FEATURE_SETTINGS['drop_cols'] + [FEATURE_SETTINGS['target']]]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=existing_cols_to_drop).fillna(0)
    y = df[FEATURE_SETTINGS['target']]
    
    # Configure selector with error handling
    selector = RFECV(
        estimator=RandomForestClassifier(
            class_weight='balanced',
            n_estimators=20,
            max_depth=5
        ),
        step=5,
        cv=2,
        min_features_to_select=10,
        n_jobs=-1
    )
    
    try:
        selector.fit(X, y)
        return X.columns[selector.support_]
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        return []

if __name__ == '__main__':
    datasets = ['crosscheck', 'studentlife', 'combined']
    
    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"Processing {dataset} dataset")
        print(f"{'='*40}")
        
        try:
            features = select_features(DATA_PATHS[f'{dataset}_processed'])
            print(f"Selected features for {dataset}:")
            print(features.tolist())
            
            pd.Series(features).to_csv(
                f"results/features/{dataset}_selected_features.csv",
                index=False
            )
            
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")