import numpy as np

"""
Class for filtering noise from the sound files using an FIR filter
"""
class FIRFilter:

    def define_highpass_filter(self, order : int = 1001, cutoff_freq : float = 700) -> np.ndarray:
        """
        Calculates the coefficients for an FIR high pass filter, the coefficients are calculated as the negative of
        a low pass filter
        :param order: the order (length) of the filter
        :param cutoff_freq: the cutoff frequency
\        :return: the coefficients
        """
        coefs = np.zeros(order)
        angular_cutoff_freq = cutoff_freq * 2 * np.pi
        for i in range(order):
            coefs[i] = np.sinc((i - order) * angular_cutoff_freq ) * (-1) ** i
        return coefs

    def filter(self, data : np.ndarray, coefs : np.ndarray) -> np.ndarray:
        """
        Filters the data using an FIR filter, it does so by doing a convolution between the coefficients and the data
        :param data: the data to be filtered
        :param coefs: the filter coefficients
        """

        return np.convolve(data, coefs, mode='same')