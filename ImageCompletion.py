import os
import numpy as np

from Parameters import Parameters
from Utility import Utility
class ImageCompletion:
    @staticmethod
    def completeLeft(spn, testSet, directory, filename):
        Utility.parseBuffer = []
        for instance in testSet:
            spn.completeLeftImage(instance)
        ImageCompletion.saveCompletionToFile(directory, filename, len(testSet))
        
    def completeBottom(spn, testSet):
        for instance in testSet:
            spn.completeBottomImage(instance)


    @staticmethod
    def saveCompletionToFile(directory, name, test_size):
        if not os.path.exists(directory):
            os.mkdir(directory)
        os.chdir(directory)
        
        length = Parameters.imageWidth * Parameters.imageHeight
        data = np.reshape(Utility.parseBuffer, (length, test_size), 'F')
        #error = np.linalg.norm(data - Utility.testSet,2) / test_size
        #print 'The mean square error is %f' % error
        np.savetxt(name, data)
        
        os.chdir('../')
  

if __name__ == '__main__':
    ImageCompletion.saveCompletionToFile('./results',None)
