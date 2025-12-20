import os
from typing import List, Dict
import pandas as pd

from utils.load_file import FileLoader
from utils.data_exploration import DataExploration

"""
Class for structuring the audio files into a dataset, preparing for the pipeline.
"""
class DatasetLoader:

    def __init__(self, root_dir: str):
        """
        Initializes the class. 
        
        Args: 
            root_dir (string): the root directory for the files. 
        
        Returns:
            Nothing. 
        """
        self.root_dir = root_dir
        self.explorer = DataExploration()

    def _list_files(self) -> List[str]:
        """
        Function to create a list of the files.
        
        Args: 
            Nothing.
        
        Returns:
            wav_paths (list of strings): A list of the file paths where the audio files are found.
        """
        wav_paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for name in filenames:
                if name.lower().endswith(".wav"):
                    wav_paths.append(os.path.join(dirpath, name))
        return wav_paths
    

    def run(self) -> pd.DataFrame:
        """
        Function to create the dataframe. It loops all WAV files, extracts features
        to build a DataFrame for analysis. 
        
        Args: 
            Nothing.
        
        Returns:
            df (Pandas DataFrame): The dataframe with features and WAV files for analysis. 
        """

        records: List[Dict[str, float | str | int]] = []

        for path in self._list_files():
            loader = FileLoader(path)
            data = loader.get_data()
            sr = loader.get_sample_rate()

            feats = self.explorer.explore(data)

            feats["file"] = path
            feats["sample_rate"] = sr

            records.append(feats)
        
        df = pd.DataFrame.from_records(records)
        return df
    