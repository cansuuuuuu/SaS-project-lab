from fast_fourier_transform import FastFourierTransform
import numpy as np
from typing import Dict

"""
Class containing the various method of spectral analysis for a discrete time series.
"""
class SpectralAnalysis:

    def __init__(self):
        self.fft_tool = FastFourierTransform()

    def _welch(self, data: np.ndarray, frame_length: int, frame_overlap: int) -> np.ndarray:
        """
        A calculation of the power spectral density of a time series. It works using welch's method, averaging out the
        frequency domain representation of overlapping frames of the time series. Each frame has the Hann window applied
        to it to avoid artifacts.
        :param data (numpy ndarray): discrete time series
        :param frame_length (int): number of samples used in each frame
        :param frame_overlap (int): number of samples that overlap between frames
        :return psd (numpy ndarray): The welch power spectral density of a time series
        """
        n = data.shape[0]
        step = int(frame_length - frame_overlap)
        rest = int((n - frame_overlap) % step)
        if rest != 0:
            padded_data = np.pad(data, step - rest)
            n = n + (step - rest)
        else:
            padded_data = data
        psd = np.zeros(frame_length)
        nb_steps = int((n - frame_overlap) // step)
        for i in range(nb_steps):
            frame = padded_data[i * step: frame_length + i * step]
            windowed_frame = frame * np.hanning(frame_length)
            fft_frame = self.fft_tool.fft(data = windowed_frame)
            psd_frame = np.square(fft_frame)
            psd = psd + psd_frame * 1 / (n // step)
        return psd


    @staticmethod
    def _spectrogram(data : np.ndarray, frame_length : int, frame_overlap : int) -> np.ndarray:
        """
        Method to draw a spectrogram, it takes the input discrete time series, applies a fft to different
        overlapping frames, with the Hanning window applied to them, puts them all into a 2d numpy ndarray.
        :param data: discrete time series
        :param frame_length: number of samples used in each frame
        :param frame_overlap: number of samples that overlap between frames
        :return: the spectrogram data for the input time series
        """
        n = data.shape[0]
        step = int(frame_length - frame_overlap)
        rest = int((n - frame_overlap) % step)
        if rest != 0:
            padded_data = np.pad(data, step - rest)
            n = n + (step - rest)
        else:
            padded_data = data
        nb_steps = int((n - frame_overlap) // step)
        spectrogram = np.zeros((frame_length, nb_steps))
        for i in range(nb_steps):
            frame = padded_data[i * step: frame_length + i * step]
            windowed_frame = frame * np.hanning(frame_length)
            fft_frame = FastFourierTransform.fft(data = windowed_frame)
            spectrogram[:, i] = fft_frame
        return spectrogram

    def spectral_analysis(self, data: np.ndarray, fs: int, frame_length: int = 2048, frame_overlap: int = 1024) -> Dict[str, float]:
        """
        Calculates various spectral qualities of the time series.
        :param data: discrete time series
        :param fs: sampling frequency of the time series
        :param frame_length: length of the FFT window
        :param frame_overlap: overlap between windows
        :return: dictionary with spectral features
        """
        psd = self._welch(data, frame_length, frame_overlap)

        psd_mean = float(np.mean(psd))

        half_n = len(psd) // 2
        freqs = np.linspace(0, fs / 2, half_n)
        psd_half = psd[:half_n]
        psd_max_freq = float(freqs[np.argmax(psd_half)])
        
        if np.sum(psd_half) == 0:
            spectral_centroid = 0.0
        else:
            spectral_centroid = float(np.sum(freqs * psd_half) / np.sum(psd_half))
            
        return {
            "psd_mean": psd_mean,
            "psd_max_freq": psd_max_freq,
            "spectral_centroid": spectral_centroid
        }