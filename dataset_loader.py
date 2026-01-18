import os
from typing import List, Dict
import pandas as pd

from load_file import FileLoader
from data_exploration import DataExploration
from fir_filter import FIRFilter

class DatasetLoader:
    """
    Class for structuring the audio files into a dataset, preparing for the pipeline.
    """
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
        self.filter = FIRFilter()
        self.frequency_response = self.filter.define_highpass_filter(cutoff_freq = 800)


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
                if name.lower().endswith((".wav", ".mp3")):
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
            if data.size == 0:
                continue

            sr = loader.get_sample_rate()
            max_samples = 5 * sr # limited to be able to process
            if data.size > max_samples:
                data = data[:max_samples]

            filtered_data = self.filter.filter(data, self.frequency_response)

            feats = self.explorer.explore(data, fs=sr)

            feats["file"] = path
            feats["sample_rate"] = sr

            label = os.path.basename(os.path.dirname(path))
            if label and label != os.path.basename(self.root_dir):
                feats["label"] = label
            else:
                continue

            records.append(feats)

        df = pd.DataFrame.from_records(records)
        return df
