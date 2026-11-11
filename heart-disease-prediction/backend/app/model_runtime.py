from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from .mc_random_forest import MultiCriterionRandomForest

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "heart_model.joblib"

class HSOStackingWrapper:
    def __init__(self, trained_models, weights):
        self.models = trained_models
        self.weights = np.asarray(weights, dtype=float)
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        probabilities = np.column_stack([model.predict_proba(X)[:, 1] for _, model in self.models])
        p = np.clip(probabilities @ self.weights, 0, 1)
        return np.column_stack([1-p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class HeartModel:
    def __init__(self):
        self.bundle = None
        self.model = None
        if ARTIFACT.exists():
            self.load()
    def load(self):
        # Older artifacts reference this class as __main__.MultiCriterionRandomForest.
        setattr(sys.modules["__main__"], "MultiCriterionRandomForest", MultiCriterionRandomForest)
        self.bundle = joblib.load(ARTIFACT)
        self.model = HSOStackingWrapper(self.bundle['trained_models'], self.bundle['best_weights'])
    @property
    def ready(self): return self.bundle is not None
    def transform(self, values):
        if not self.ready: raise RuntimeError('Model artifacts are missing. Run python train_model.py after placing heart.csv in backend/data/.')
        b = self.bundle
        features = b['input_features']
        missing = [feature for feature in features if feature not in values]
        if missing:
            raise ValueError(f"Missing required model inputs: {', '.join(missing)}")
        row = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features,
        )
        row = row.apply(pd.to_numeric, errors='coerce').fillna(
            pd.Series(b['imputer_statistics'], index=features)
        )
        scaled = b['scaler'].transform(row.to_numpy())
        if 'kmeans' in b:
            distances = b['kmeans'].transform(scaled)
            cluster = b['kmeans'].predict(scaled)
            enriched = np.column_stack([scaled, cluster, distances])
            selected = enriched[:, b['selected_indices']]
            names = b['selected_feature_names']
            return pd.DataFrame(selected, columns=names)
        return pd.DataFrame(scaled, columns=b.get('selected_feature_names', features))
    def predict(self, values):
        X = self.transform(values)
        p = float(self.model.predict_proba(X)[0, 1])
        return {'model_score': round(p, 4), 'class_prediction': int(p >= .5), 'selected_features': X.iloc[0].to_dict()}

heart_model = HeartModel()
