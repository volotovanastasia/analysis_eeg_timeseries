import numpy as np
from parameters.Parameter import Parameter


class BoxCountingDimension(Parameter):
    def __init__(self, box_size=0.001):
        self.box_size = box_size

    def calculate(self, data) -> float:
        x_values, y_values = data, data
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        x_min -= self.box_size
        y_min -= self.box_size
        x_max += self.box_size
        y_max += self.box_size

        x_range = x_max - x_min
        y_range = y_max - y_min

        num_boxes_x = int(x_range / self.box_size) + 1
        num_boxes_y = int(y_range / self.box_size) + 1

        grid = np.zeros((num_boxes_x, num_boxes_y))

        for i in range(len(x_values)):
            x_idx = int((x_values[i] - x_min) / self.box_size)
            y_idx = int((y_values[i] - y_min) / self.box_size)

            if 0 <= x_idx < num_boxes_x and 0 <= y_idx < num_boxes_y:
                grid[x_idx, y_idx] = 1
            else:
                print(f"Out of bounds: x_idx={x_idx}, y_idx={y_idx}")

        filled_boxes = np.sum(grid)
        dimension = -np.log(filled_boxes) / np.log(self.box_size)

        return dimension
