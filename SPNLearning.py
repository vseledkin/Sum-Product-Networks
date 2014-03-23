from SPN import SPN
from Parameters import Parameters
from Dataset import Dataset
from Utility import Utility
from ImageCompletion import ImageCompletion
import numpy as np
import time
class SPNLearning:
    """
        Learning Sum-Product Networks by using hard EM
        and hard Gradient Descent
    """
    def __init__(self):
        self.__spn = SPN()

    def saveModel(self):
        self.__spn.saveModel()

    def getLearnedSPN(self):
        return self.__spn
    
    def learn(self, dataset, mode = 'EM'):
        if mode == 'EM':
            self.__learnByHardEM(dataset)
        elif mode == 'GD':
            self.__learnByGradientDescent(dataset)

    def __learnByHardEM(self, dataset):
        # initialize SPN
        print 'Initializing...'
        self.__spn.addTrainingSet(dataset)
        self.__spn.initialize()

        # learning in minibatches
        prior = Parameters.prior
        data_size = self.__spn.getTrainingSetSize()
        batchSize = Parameters.batchSize

        old_likelihood = 0.0
        start = time.time()
        for iteration in xrange(1, Parameters.maxIteration + 1):
            print 'Iteration %d' % iteration
            if iteration < 10:
                Parameters.prior = prior * iteration / 10
                
            for batchIndex in xrange(0, data_size / batchSize):
                lower = batchIndex * batchSize
                upper = (batchIndex + 1) * batchSize
                upper = upper if upper < data_size else data_size

                # Using the incremental EM from (Neal and Hinton)
                # first of all, remove the updates from previous iteration
                print 'Clear parse from previous iteration'
                for index in xrange(lower, upper):
                    self.__spn.clearParseToMAP(index)
                self.__spn.clearParseToMAPFromBuffer()
                assert len(Utility.parseBuffer) == 0
                # then, doing MAP inference for the new iteration 
                # E-step for this minibatch
                print 'E-step in minibatch %d' % batchIndex
                for index in xrange(lower, upper):
                    print '...MAP inference for %d-th instance' % index 
                    instance = self.__spn.getTrainingInstance(index)
                    # This will find the MAP SumNode and best
                    # decomposition of the regions
                    self.__spn.MAPinference(index, instance)
                    # This may create ProdNode for each decomposition
                    self.__spn.setParseToMAP(index)
                    
                # M-step for this minibatch
                # This will append ProdNodes to its parent SumNode
                print 'M-step in minibatch %d' % batchIndex
                self.__spn.setParseToMAPFromBuffer()
                assert len(Utility.parseBuffer) == 0
                # for test
                print self.__spn._SPN__rootSumNode.getCounts()
                for node in self.__spn._SPN__rootSumNode.getChildren():
                    print self.__spn._SPN__rootSumNode.getChildCounts(node)
            # after sweeping through the whole dataset, clear unusued
            print 'Clear unused decompositions'
            self.__spn.clearUnusedDecomp()

            # check convergence
            likelihood = 0.0
            for instance in dataset:
                likelihood += self.__spn.getLogLikelihood(instance)
            print 'The likelihood is %f' % likelihood
            if iteration == 1:
                old_likelihood = likelihood
            else:
                diff = np.abs(likelihood - old_likelihood)
                print diff
                if diff < Parameters.thresholdForConvergence:
                    print 'The algorithm has converged!'
                    break
                else:
                    old_likelihood = likelihood
        end = time.time()
        print 'The training time is %f' % (end - start)
            
if __name__ == '__main__':
    s = SPNLearning()
    data = Dataset()
    data.loadData('olivetti.raw')
    s.learn(data.getTrainingSet())
    spn = s.getLearnedSPN()
    ImageCompletion.completeLeft(spn, data.getTestSet(), './results/', 'demo.dat')
