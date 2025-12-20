import numpy as np

from utils.load_file import FileLoader
from utils.dataset_loader import DatasetLoader
from utils.data_exploration import DataExploration

def main():
    root_dir = "project_2/dataset"
    loader = DatasetLoader(root_dir)
    df = loader.run()
    
    print(df.head())
    df.to_csv("project_2_features.csv", index=False)

if __name__ == "__main__":
    main()
