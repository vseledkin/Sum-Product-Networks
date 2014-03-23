import numpy as np


class Node:
    """
        Generic Node of SPN
    """
    ZERO = float('-inf')
    def __init__(self):
        self.__logValue = Node.ZERO
        self.__logDerivative = Node.ZERO

    def getLogDerivative(self):
        return self.__logDerivative

    def setLogDerivative(self, value):
        self.__logDerivative = value
    
    def getLogValue(self):
        return self.__logValue

    def setLogValue(self, value):
        """
            The value is stored in log scale
        """
        self.__logValue = value

    def evaluate(self):
        raise NotImplementedError("This is an abstract method.")

    def passDerivative(self):
        raise NotImplementedError("This is an abstract method.")
