# code/evaluation/8_performance_evaluation.py
import sys
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATHS, MODEL_PATHS, BASE_DIR

def loso_evaluation():
    results = []
    
    for dataset in ['crosscheck', 'studentlife', 'combined']:
        try:
            df = pd.read_csv(DATA_PATHS[f'{dataset}_processed'])
            model_path = MODEL_PATHS[dataset]/('combined_model.pkl' if dataset == 'combined' else 'best_model.pkl')
            
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                continue
                    
            model = joblib.load(model_path)
            subjects = df['study_id'].unique()
            
            for subject in subjects:
                train = df[df['study_id'] != subject]
                test = df[df['study_id'] == subject]
                
                if len(test) < 3 or len(train) < 10:
                    continue
                    
                X_train = train.drop(columns=['ema_score', 'study_id'], errors='ignore')
                y_train = train['ema_score']
                X_test = test.drop(columns=['ema_score', 'study_id'], errors='ignore')
                y_test = test['ema_score']
                
                # Skip single-class folds
                if len(y_train.unique()) < 2:
                    continue
                
                # Check minority class size
                y_counts = y_train.value_counts()
                if y_counts.min() < 4:  # k_neighbors=3 needs at least 4 samples
                    continue
                    
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    results.append({
                        'dataset': dataset,
                        'subject': subject,
                        'accuracy': accuracy_score(y_test, preds),
                        'f1': f1_score(y_test, preds, average='weighted')
                    })
                except Exception as e:
                    print(f"Subject {subject} error: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Dataset {dataset} error: {str(e)}")
            continue
    
    # Save results
    results_dir = BASE_DIR/'results/metrics'
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_dir/'loso_results.csv', index=False)
    print("Evaluation completed. Results saved to:", results_dir/'loso_results.csv')

if __name__ == '__main__':
    loso_evaluation()
