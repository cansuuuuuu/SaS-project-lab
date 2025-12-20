import wave
import numpy as np
import librosa

"""
Class in charge of loading a individual files from the dataset.
It also returns any features of the sound file that are required by other classes, such as sample rate or amplitude.
"""
class FileLoader:

    def __init__(self, filepath: str):
        """
        Initializer for the class.
        
        Args: 
            filepath (string): the filepath of the file we will load
        
        Returns:
            Nothing
        """
        self._filepath = filepath
        self._sample_rate: int = 0
        self._data: np.ndarray = np.array([])
        self._load_audio()

    
    def _load_audio(self):
        """
        Function for loading audio files, supporting both WAV and MP3.
        The "data" variable loaded by librosa is a Numpy floating point time series
        which is normalized to the range of -1.0 to 1.0

        Args: 
            Nothing.
        
        Returns:
            Nothing.
        """
        data, sr = librosa.load(self._filepath, sr=None, mono=True)
        self._sample_rate = int(sr)
        self._data = (data * 32767).astype(np.int16)
    
    def get_sample_rate(self) -> int:
        """
        Function to get the sample rate of the audio file.
        
        Args: 
            Nothing.
        
        Returns:
            The sample rate, calculated in previous function.
        """
        return self._sample_rate
    
    def get_data(self) -> np.ndarray:
        """
        Returns the floating pount time series of audio files     
        Args: 
            Nothing.
        
        Returns:
            The audio sample as the floating point time series in NumPy. 
        """
        return self._data.copy()