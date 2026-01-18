import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset

from dataset_loader import DatasetLoader
from classifier import BirdClassifier
import os

def main():
    root_dir = "dataset/train-bird-audio"

    if os.path.exists("dataset.joblib"):
        df = joblib.load("dataset.joblib")
        print("Dataset loading from dataset.joblib")
    else:
        loader = DatasetLoader(root_dir)
        df = loader.run()
        joblib.dump(df, "dataset.joblib")

    print(f"Loaded {len(df)} samples from {df['label'].nunique()} classes.")

    feature_cols = [
        "rms", "std", "avg_amplitude", "min_amplitude", "max_amplitude", "crest_factor",
        "psd_mean", "psd_max_freq", "spectral_centroid"
    ]
    x = df[feature_cols]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y)

    if os.path.exists('bird_model.joblib'):
        classifier = joblib.load('bird_model.joblib')
        print("Model loaded from bird_model.joblib")
    else:
        classifier = BirdClassifier(feature_names=feature_cols)
        print("Training Random Forest classifier...")
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, 'bird_model.joblib')

    y_pred = classifier.predict(x_test)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    classifier.explain(x[:50])

    df.to_csv("project_1_features.csv", index=False)
    print("\nFeatures saved to project_1_features.csv")

if __name__ == "__main__":
    main()
