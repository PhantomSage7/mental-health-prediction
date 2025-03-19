# code/modeling/model_utils.py
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np
from imblearn.combine import SMOTETomek

class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)
        
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
        
    def transform(self, X):
        clusters = self.model.predict(X)
        return np.hstack([X, clusters.reshape(-1, 1)])

def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote_tomek', SMOTETomek(k_neighbors=3)),  # Reduced from 5 to 3
        ('cluster', ClusterTransformer(n_clusters=3)),
        ('classifier', model)
    ])
