import numpy as np
from Parameters import Parameters
class Instance:
    """
        Represent a raw image
    """
    def __init__(self):
        self.width = Parameters.imageWidth
        self.height = Parameters.imageHeight
        self.__values = np.array((self.width, self.height))
        self.__mean = 0.0
        self.__std = 0.0

    def setMean(self, value):
        self.__mean = value

    def setStd(self, value):
        self.__std = value

    def getMean(self):
        return self.__mean

    def getStd(self):
        return self.__std

    def getValue(self, row, col):
        assert row < self.width and col < self.height
        return self.__values[row, col]

    def setValue(self, row, col, value):
        assert row < self.width and col < self.height
        self.__values[row, col] = value

    def setValue(self, mat):
        row, col = mat.shape
        assert row == self.width and col == self.height
        self.__values = mat
    
        
