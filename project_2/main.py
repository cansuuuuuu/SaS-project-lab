from utils.dataset_loader import DatasetLoader

def main():
    root_dir = "dataset"
    loader = DatasetLoader(root_dir)
    df = loader.run()

    print(df.head())
    df.to_csv("project_2_features.csv", index=False)

if __name__ == "__main__":
    main()
