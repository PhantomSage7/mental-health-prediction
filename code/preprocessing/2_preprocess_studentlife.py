import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATHS

def preprocess_studentlife():
    df = pd.read_csv(DATA_PATHS['studentlife_raw'], parse_dates=['day'])
    
    # Convert datetime to numerical features
    df['day_of_week'] = df['day'].dt.dayofweek
    df['day_hour'] = df['day'].dt.hour
    df = df.drop(columns=['day'])  # Remove original datetime column
    
    # Target creation with explicit type handling
    stress_filled = df['ema_Stress_level'].fillna(df['ema_Stress_level'].median())
    mood_filled = df['ema_Mood_sad'].fillna(df['ema_Mood_sad'].median())
    df['ema_score'] = (stress_filled * 2 + mood_filled).round().astype(int)
    
    # Handle zero scores and ensure integer type
    df['ema_score'] = df['ema_score'].replace(0, 1).astype(int)
    df = df.dropna(subset=['ema_score'])
    
    # Save preprocessed data
    df.to_csv(DATA_PATHS['studentlife_processed'], index=False)

    # Print summary
    print("Processed StudentLife Data Summary:")
    print(df.info())
    print(df.describe())

if __name__ == '__main__':
    preprocess_studentlife()