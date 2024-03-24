from abc import ABC, abstractmethod


class Parameter(ABC):
    @abstractmethod
    def calculate(self, data) -> float:
        pass
