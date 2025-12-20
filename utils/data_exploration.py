import numpy as np
from typing import List, Dict

"""
Class for exploring the data. 
Here we get the root mean square, crest factor, standard deviation, 
average amplitude and minimum/maximum amplitude of the data. 
"""
class DataExploration:

    def _rms(self, signal: np.ndarray) -> float:
        """
        Function to calculate the Root Mean Square (RMS) of a given signal.
        The root mean square is calculated by finding the mean of the squared signal, 
        and then finding the square root of that value. 
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file
        
        Returns:
            rms (float): the root mean square of the signal.
        """
        rms = float(np.sqrt(np.mean(signal**2)))
        return rms


    def _std(self, signal: np.ndarray) -> float:
        """
        Function to calculate the standard deviation of a given signal. 
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file
        
        Returns:
            std (float): the standard deviation of the signal.
        """
        std = float(np.std(signal))
        return std


    def _avg_amplitude(self, signal: np.ndarray) -> float:
        """
        This function calculates the average amplitude of a given signal.
        We use the NumPy mean function and calculate the mean of the absolute values
        of the signal, this gives us the average amplitude.  
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file
        
        Returns:
            avg_amplitude (float): average amplitude of the signal.
        """
        avg_amplitude = float(np.mean(np.abs(signal)))
        return avg_amplitude


    def _min_max(self, signal: np.ndarray) -> tuple[float, float]:
        """
        This function calcualtes the minimum and maximum amplitude of a given signal
        using the NumPy functions min and max.  
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file
        
        Returns:
            min_max (tuple with floats): tuple with the minimum and maximum amplitudes of the signal.
        """
        min_max = float(np.min(signal)), float(np.max(signal))
        return min_max


    def _crest_factor(self, signal: np.ndarray, rms: float) -> float:
        """
        Function for calculating the crest factor of a signal. The calculation
        is done by dividing the peak of the signal with the root mean square.  
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file
            rms (float): Root Mean Square of the signal 
        
        Returns:
            cf (float): Crest factor of the signal. 
        """
        if rms == 0.0:
            return 0.0
        peak = float(np.max(np.abs(signal)))
        cf = peak / rms
        return cf


    def explore(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Function to do the data exploration on the data. 
        The function creates a dictionary of all the features extracted from the signal.
        
        Args: 
            signal (np,ndarray): The signal data extracted from the file.
        
        Returns:
            explored_data (Dict[str, float]): a dictionary containing the RMS, STD, 
                avg, min, max amplitude and the crest_factor of the signal
        """

        rms = self._rms(signal)
        std = self._std(signal)
        avg_amp = self._avg_amplitude(signal)
        min_amp, max_amp = self._min_max(signal)
        crest = self._crest_factor(signal, rms)

        explored_data = {
            "rms": rms,
            "std": std,
            "avg_amplitude": avg_amp,
            "min_amplitude": min_amp,
            "max_amplitude": max_amp,
            "crest_factor": crest,
        }

        return explored_data
