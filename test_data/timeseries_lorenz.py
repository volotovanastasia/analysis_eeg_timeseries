from LorenzAttractor import lorenz_attractor
import numpy as np
import matplotlib.pyplot as plt

# Параметры аттрактора Лоренца
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
num_steps = 5000  # Количество шагов
dt = 0.01  # Величина шага времени

# Генерируем аттракторы Лоренца

plt.figure(figsize=(10, 6))
for i in range(5):
    x0, y0, z0 = np.random.rand(3) * 20 - 10
    x_values, _, _ = lorenz_attractor(sigma, rho, beta, x0, y0, z0, num_steps, dt)
    plt.plot(x_values[:1000], label=f'Временной ряд {i + 1}')

plt.xlabel('Время')
plt.ylabel('Значение')
plt.title('Первые 1000 точек временных рядов на основе аттрактора Лоренца')
plt.legend()
plt.grid(True)

plt.show()
