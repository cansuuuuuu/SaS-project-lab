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

    def fit(self, x, y):
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, x):
        if not self.is_fitted:
            raise ValueError("Fit model first!")
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def explain(self, x: np.ndarray):
        if not self.is_fitted:
            raise ValueError("Fit first!")
        x_scaled = self.scaler.transform(x)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(x_scaled)
        shap.summary_plot(shap_values, x_scaled, plot_type="bar")
        return shap_values