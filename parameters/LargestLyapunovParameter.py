from parameters.Parameter import Parameter
import numpy as np


class LargestLyapunovExponent(Parameter):
    def __init__(self, embedding_dimension, time_delay):
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay

    def calculate(self, data) -> float:
        lyapunov_exponents = self._compute_largest_lyapunov_exponent(data)
        return np.mean(lyapunov_exponents)

    def _time_delay_embedding(self, data):
        n = len(data)
        m = self.embedding_dimension
        tau = self.time_delay
        embedded_data = np.zeros((n - (m - 1) * tau, m))

        for i in range(m):
            embedded_data[:, i] = data[i * tau: i * tau + len(embedded_data)]

        return embedded_data

    def _compute_largest_lyapunov_exponent(self, data):
        embedded_data = self._time_delay_embedding(data)
        n = len(embedded_data)
        m = self.embedding_dimension
        lyapunov_exponents = np.zeros(n - 1)

        for i in range(1, n):
            # Вычисляем производные по всем осям для каждой строки фазового портрета
            gradients = np.gradient(embedded_data[:i + 1], axis=0)

            # Формируем матрицу Якоби из производных
            J = np.column_stack([gradients[j][-1] for j in range(m)])

            # Вычисляем собственные значения матрицы Якоби
            eigenvalues, _ = np.linalg.eig(J)
            largest_real_part = np.max(np.real(eigenvalues))
            lyapunov_exponents[i - 1] = largest_real_part

        return lyapunov_exponents


# data = np.random.randn(100)
#
# embedding_dimension = 1
# time_delay = 1
#
# lyapunov_calculator = LargestLyapunovExponent(embedding_dimension, time_delay)
# lyapunov_exponent = lyapunov_calculator.calculate(data)
#
# print(f"Старший показатель Ляпунова: {lyapunov_exponent}")
