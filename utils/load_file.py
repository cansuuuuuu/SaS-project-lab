import wave
import numpy as np

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
        self._load_wav()

    
    def _load_wav(self):
        """
        Function for actually loading in the files.
        It opens the WAV audio files using wave.open from wave library.
        It then reads and stores the metadata (such as sample rate, width and channels).
        It also stores the raw audio data and converts it into a Numpy array, and reshapes the
        audio from stereo to mono. The conversion to mono preservers singal content for
        time-domain analysis.

        Args: 
            Nothing.
        
        Returns:
            Nothing.
        """
        with wave.open(self._filepath, 'rb') as wav_file:
            self._sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            frames = wav_file.readframes(wav_file.getnframes())
            self._data = np.frombuffer(frames, dtype=np.int16)
            
            if n_channels > 1:
                self._data = self._data.reshape(-1, n_channels).mean(axis=1)
    
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
        Returns the 16-bit audio sample as NumPY int16 array.
        
        Args: 
            Nothing.
        
        Returns:
            The audio sample as a NumPy array, also calculated in previous function.
        """
        return self._data.copy()