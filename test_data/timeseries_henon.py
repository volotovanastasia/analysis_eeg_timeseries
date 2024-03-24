from HenonAttractor import HenonAttractor
import numpy as np
import matplotlib.pyplot as plt
import random
from ordpy import renyi_entropy

num_steps = 10000

for i in range(5):
    x0 = random.uniform(0, 0.1)
    y0 = random.uniform(0, 0.1)
    henon = HenonAttractor(x0, y0)
    x_values, y_values = henon.generate_attractor(num_steps)
    print(renyi_entropy(x_values))

