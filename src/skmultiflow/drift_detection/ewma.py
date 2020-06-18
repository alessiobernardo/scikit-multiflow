from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import  math


class EWMA(BaseDriftDetector):
    """ Drift detection method based on EWMA [1]

        Parameters
        ----------
        delta : float (default=0.2)
            The delta parameter for the ADWIN algorithm.

        References
        ----------
        .. [1] Ross, Gordon J., et al. "Exponentially weighted moving average charts
            for detecting concept drift." Pattern recognition letters 33.2 (2012): 191-198.

        Examples
        --------
        >>> # Imports
        >>> import numpy as np
        >>> from skmultiflow.drift_detection import EWMA
        >>> ewma = EWMA()
        >>> # Simulating a data stream as a normal distribution of 1's and 0's
        >>> data_stream = np.random.randint(2, size=2000)
        >>> # Changing the data concept from index 999 to 2000
        >>> for i in range(999, 2000):
        ...     data_stream[i] = np.random.randint(4, high=8)
        >>> # Adding stream elements to ADWIN and verifying if drift occurred
        >>> for i in range(2000):
        ...     ewma.add_element(data_stream[i])
        ...     if ewma.detected_change():
        ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))

    """

    def __init__(self, delta=.2):
        super().__init__()
        self.delta = delta
        self.p_0 = 0
        self.p_sum = 0
        self.t = 1
        self.std_z = 0
        self.z = 0
        super().reset()

    def add_element(self, input_value):
        if self.in_concept_change:
            self.__init__(delta=self.delta)

        self.p_sum += input_value
        self.p_0 = self.p_sum / self.t
        self.std_z = math.sqrt(self.p_0 * (1.0 - self.p_0) * self.delta * \
                               (1.0 - math.pow(1.0 - self.delta, 2.0 * self.t)) / (2.0 - self.delta))
        self.t += 1
        self.z = self.delta * input_value + (1.0 - self.delta) * self.z
        l_t = self.__get_polynomial_approximation(self, p_0=self.p_0)

        self.estimation = self.p_0
        self.in_concept_change = False
        self.in_warning_zone = False

        if self.z > self.p_0 + 0.5 * l_t * self.std_z:
            self.in_warning_zone = True
        if self.z > self.p_0 + l_t * self.std_z:
            self.in_concept_change = True
            self.in_warning_zone = False

    @staticmethod
    def __get_polynomial_approximation(self, p_0):
        # return 2.76 - 6.23 * p_0 + 18.12 * math.pow(p_0, 3) - 312.45 * math.pow(p_0, 5) + 1002.18 * math.pow(p_0, 7)
        return 3.97 - 6.56 * p_0 + 48.73 * math.pow(p_0, 3) - 330.13 * math.pow(p_0, 5) + 848.18 * math.pow(p_0, 7)
        # return 1.17 + 7.56 * p_0 - 21.24 * math.pow(p_0, 3) + 112.12 * math.pow(p_0, 5) - 987.23 * math.pow(p_0, 7)
