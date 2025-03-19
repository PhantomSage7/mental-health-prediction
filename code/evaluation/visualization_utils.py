#code\evaluation\visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from config import BASE_DIR
import pandas as pd

def plot_results(results_path):
    df = pd.read_csv(results_path)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dataset', y='accuracy', data=df)
    plt.title('Model Accuracy Comparison')
    plt.savefig(BASE_DIR/'results/figures/accuracy_comparison.png')
    plt.close()