#code\feature_analysis\4_feature_comparison.py
import sys
import os
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_PATHS

# Define BASE_DIR as the project root
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def feature_comparison():
    cc = pd.read_csv(DATA_PATHS['crosscheck_processed'])
    sl = pd.read_csv(DATA_PATHS['studentlife_processed'])
    
    common_features = list(set(cc.columns) & set(sl.columns))
    
    results = []
    for feat in common_features:
        statistic, pvalue = ks_2samp(cc[feat].dropna(), sl[feat].dropna())
        results.append({
            'feature': feat,
            'ks_statistic': statistic,
            'p_value': pvalue
        })
    
    result_df = pd.DataFrame(results)
    
    # Ensure the directory exists before saving the file
    output_path = BASE_DIR / 'results' / 'metrics'
    output_path.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_path / 'feature_comparison.csv', index=False)

if __name__ == '__main__':
    feature_comparison()
