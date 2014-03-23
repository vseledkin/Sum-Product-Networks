import numpy as np
from Utility import Utility
from Parameters import Parameters
from Instance import Instance
class Dataset():
    """
        The Olivetti dataset have 400 pictures and
        each of the pic is 64 x 64 pixels and the
        whole dataset is organized as a 4096 x 400
        matrix. And for each column the pic is stacked
        in row-wise order to a long vector of size 4096.
    """
    def __init__(self):
        self.directory = './data/olivetti/'
        self.width = Parameters.imageWidth
        self.height = Parameters.imageHeight
        self.__train = []
        self.__test = []
        
    def loadData(self, filename):
        ds = np.loadtxt(self.directory+filename)
        self.__instances = self.__createInstances(ds)
        train_idx, test_idx = self.__splitDataset(self.__instances)
        self.__train = self.__instances[train_idx]
        self.__test = self.__instances[test_idx]
        Utility.testSet = ds[:,test_idx]
        
    def getTrainingSet(self):
        return self.__train
    
    def getTestSet(self):
        return self.__test

    def __createInstances(self, dataset):
        """
            Create all the instances of a dataset
            and a list of all the instances
        """
        rows, cols = dataset.shape
        inst = []
        for col in range(cols):
            tmp, mean, std = self.__standardise(dataset[:, col])
            pic = tmp.reshape(self.width, self.height, order='F')
            instance = Instance()
            instance.setValue(pic)
            instance.setMean(mean)
            instance.setStd(std)
            inst.append(instance)
        return np.array(inst)
    def __splitDataset(self, instances):
        """
            Split all the instances into training instances
            and test instances, then return their corresponding
            indices
        """
        size = len(instances)
        train_idx = np.arange(0, size-Parameters.testSetSize)
        test_idx = np.arange(size-Parameters.testSetSize, size)
        return train_idx, test_idx
        
    def __standardise(self, col):
        img = (col - np.mean(col)) / np.std(col)
        return img, np.mean(col), np.std(col)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
##    d = Dataset()
##    d.loadData('olivetti.raw')
##    print d.getTestSet()[0].getValue(1,1)
    a = np.loadtxt('./results/demo.dat')
    
    img = a[:,20].reshape(64,64,order='F')

    #rimg = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    #im = Image.fromarray(rimg)
    #im.save('demo.png')
    plt.imshow(img)
    plt.gray()
    plt.show()
    #d = a[:,0]
    #d = (d - np.mean(d)) / np.std(d)
    #print d
