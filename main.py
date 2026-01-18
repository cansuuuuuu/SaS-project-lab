from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from dataset_loader import DatasetLoader
from classifier import BirdClassifier

def main():
    root_dir = "dataset/train-bird-audio"
    loader = DatasetLoader(root_dir)
    df = loader.run()

    print(f"Loaded {len(df)} samples from {df['label'].nunique()} classes.")

    feature_cols = [
        "rms", "std", "avg_amplitude", "min_amplitude", "max_amplitude", "crest_factor",
        "psd_mean", "psd_max_freq", "spectral_centroid"
    ]
    x = df[feature_cols]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    classifier = BirdClassifier(feature_names=feature_cols)
    print("Training Random Forest classifier...")
    classifier.fit(x_train, y_train)

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
