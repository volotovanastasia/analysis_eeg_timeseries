import numpy as np


class LorenzAttractor:
    def __init__(self, sigma=10, rho=28, beta=8/3, x0=0, y0=1, z0=1.05):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    def generate_attractor(self, num_steps, dt):
        x_values = np.zeros(num_steps)
        y_values = np.zeros(num_steps)
        z_values = np.zeros(num_steps)

        x_values[0] = self.x0
        y_values[0] = self.y0
        z_values[0] = self.z0

        for i in range(1, num_steps):
            x_dot = self.sigma * (y_values[i-1] - x_values[i-1])
            y_dot = x_values[i-1] * (self.rho - z_values[i-1]) - y_values[i-1]
            z_dot = x_values[i-1] * y_values[i-1] - self.beta * z_values[i-1]

            x_values[i] = x_values[i-1] + x_dot * dt
            y_values[i] = y_values[i-1] + y_dot * dt
            z_values[i] = z_values[i-1] + z_dot * dt

        return x_values, y_values, z_values

# # Example usage:
# sigma = 10
# rho = 28
# beta = 8/3
# num_steps = 10000
# dt = 0.01
# x0 = 0
# y0 = 1
# z0 = 1.05
#
# lorenz = LorenzAttractor(sigma, rho, beta, x0, y0, z0)
# x_values, y_values, z_values = lorenz.generate_attractor(num_steps, dt)
