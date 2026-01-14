from .fast_fourier_transform import FastFourierTransform
import numpy as np

"""
Class containing the various method of spectral analysis for a discrete time series.
"""
class SpectralAnalysis:


    @staticmethod
    def _welch(data: np.ndarray, frame_length: int, frame_overlap: int) -> np.ndarray:
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
        for i in range((n - frame_overlap) // step):
            frame = padded_data[i * step: frame_length + i * step]
            windowed_frame = frame * np.hanning(frame_length)
            fft_frame = FastFourierTransform.fft(windowed_frame)
            psd_frame = np.square(fft_frame)
            psd = psd + psd_frame * 1 / (n // step)
        return psd


    @staticmethod
    def _spectrogram(data : np.ndarray, fs : int, ) -> np.ndarray:
