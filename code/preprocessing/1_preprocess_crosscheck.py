#code\preprocessing\1_preprocess_crosscheck.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_PATHS
import pandas as pd
import numpy as np


def preprocess_crosscheck():
    df = pd.read_csv(DATA_PATHS['crosscheck_raw'])
    
    # Handle missing columns safely
    df = df.drop(columns=['missing_days', 'quality_activity'], errors='ignore')
    
    # Handle missing data
    if 'sleep_duration' in df.columns:
        df['sleep_duration'] = df['sleep_duration'].fillna(df['sleep_duration'].median())
    if 'ema_resp_time_median' in df.columns:
        df['ema_resp_time_median'] = df['ema_resp_time_median'].fillna(df['ema_resp_time_median'].mean())

    # Temporal features (handle invalid timestamps)
    if 'sleep_start' in df.columns:
        df['sleep_start_hour'] = pd.to_datetime(df['sleep_start'], errors='coerce').dt.hour
    if 'sleep_end' in df.columns:
        df['sleep_end_hour'] = pd.to_datetime(df['sleep_end'], errors='coerce').dt.hour

    # Normalization (sum only existing loc_dist_ep_* columns)
    df['loc_dist_total'] = df.filter(like='loc_dist_ep_').sum(axis=1)
    
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df = df[numeric_cols]
    
    
    # Save processed data
    df.to_csv(DATA_PATHS['crosscheck_processed'], index=False)

    # Print processing summary
    print("\n✅ Data preprocessing completed successfully.\n")
    print("Processed Data Info:")
    print(df.info())
    print("\nProcessed Data Statistics:")
    print(df.describe())


if __name__ == '__main__':
    preprocess_crosscheck()
