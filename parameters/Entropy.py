from parameters.Parameter import Parameter
import ordpy


class Entropy(Parameter):
    def __init__(self):
        pass

    def calculate(self, data) -> float:
        return ordpy.renyi_entropy(data)
