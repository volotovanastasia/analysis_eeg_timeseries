import random

class HenonAttractor:
    def __init__(self, x0, y0, a=1.4, b=0.3):
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b

    def generate_attractor(self, n):
        x_values = [self.x0]
        y_values = [self.y0]

        for i in range(1, n):
            x = y_values[i - 1] + 1 - self.a * x_values[i - 1] ** 2
            y = self.b * x_values[i - 1]
            x_values.append(x)
            y_values.append(y)

        return x_values, y_values

a = 1.4
b = 0.3
n = 200000
x0 = random.uniform(0, 0.1)
y0 = random.uniform(0, 0.1)

henon = HenonAttractor(x0, y0, a, b)
x_values, y_values = henon.generate_attractor(n)
