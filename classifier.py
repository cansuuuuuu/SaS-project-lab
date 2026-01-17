from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np

class BirdClassifier:

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Fit model first!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def explain(self, X: np.ndarray):
        if not self.is_fitted:
            raise ValueError("Fit first!")
        X_scaled = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        shap.summary_plot(shap_values, X_scaled, X, feature_names=self.feature_names, show=False)
        return shap_values