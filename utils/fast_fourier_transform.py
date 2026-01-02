import numpy as np

class FastFourierTransform:

    def _recursive_dft(self,sequence: np.ndarray) -> np.ndarray:
        n = sequence.shape[0]

        if n <= 1:
            return sequence

        even = self._recursive_dft(sequence[::2])
        odd = self._recursive_dft(sequence[1::2])

        odd = np.array([np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)])

        first_half = []
        second_half = []

        for k in range(n // 2):
            first_half.append(even[k] + odd[k])
            second_half.append(even[k] - odd[k])

        first_half = np.array(first_half)
        second_half = np.array(second_half)

        return np.concatenate((first_half, second_half))

    def fft(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        if n & n - 1 != 0:
            next_power_of_2 = int(np.ceil(np.log2(n)))
            padded_data = np.pad(data, (0, int(2 ** next_power_of_2 - n)))
        else:
            padded_data = data
        return self._recursive_dft(padded_data)

