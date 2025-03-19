#code\preprocessing\3_combine_datasets.py
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATHS

def combine_datasets():
    cc = pd.read_csv(DATA_PATHS['crosscheck_processed'])
    sl = pd.read_csv(DATA_PATHS['studentlife_processed'])
    
    # Align feature names
    cc = cc.rename(columns={'ema_neg_score': 'negative_score'})
    sl = sl.rename(columns={'ema_Stress_level': 'stress_score'})
    
    # Common features
    common_cols = list(set(cc.columns) & set(sl.columns))
    combined = pd.concat([cc[common_cols], sl[common_cols]], axis=0)

    # Add after concatenation: Clean and process 'ema_score'
    combined = combined.dropna(subset=['ema_score'])
    combined['ema_score'] = combined['ema_score'].round().astype(int)
    
    # Save processed dataset
    combined.to_csv(DATA_PATHS['combined'], index=False)
    combined.to_csv(DATA_PATHS['combined_processed'], index=False)

    print(f"Common features ({len(common_cols)}):")
    print(sorted(common_cols))
    print("Final dataset summary:")
    print(combined.info())

if __name__ == '__main__':
    combine_datasets()
