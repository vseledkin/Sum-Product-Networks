from Parameters import Parameters
from Region import Region
from Instance import Instance
from Dataset import Dataset
from Utility import Utility
from Decomposition import Decomposition
from Node import Node
import numpy as np
class SPN():
    """
        The main interface of Sum-Product Networks
    """
    ###########################################################                    
    #                 SPN Learning part                       #
    ###########################################################
    def __init__(self):
        """
            We will work on multi-resolution in SPN, in hight
            level we will work with patches larger than 4 x 4
            pixelsand and in low level we will work with the 4 x 4
            pixels patches directly.
        """
        self.__coarseRowStep = \
                Parameters.imageWidth / Parameters.baseResolution
        self.__coarseColStep = \
                Parameters.imageHeight / Parameters.baseResolution
        # This is used to remember all the segmented region id in
        # order to avoid those clunky for-loop in other functions
        self.__coarseRegionId = []
        self.__fineRegionId = []
        self.__pixelRegionId = []
        
    def __initUnitRegion(self, r):
        # only region for a pixel will have means, variances
        r.allocateSumNodes(Parameters.numSumNodePerPixel)
        r.means = np.zeros(Parameters.numSumNodePerPixel)
        r.variances = np.zeros(Parameters.numSumNodePerPixel)
        r.counts = np.zeros(Parameters.numSumNodePerPixel)
        
        num_inst = len(self.__dataset)
        quantileSize = num_inst / Parameters.numSumNodePerPixel
        
        values = np.zeros(num_inst)
        for i, instance in enumerate(self.__dataset):
            values[i] = instance.getValue(r.rowUp, r.columnLeft)
        values = np.sort(values)
        for step in xrange(Parameters.numSumNodePerPixel):
            lowerIndex = step * quantileSize
            upperIndex = (step + 1) * quantileSize
            upperIndex = upperIndex if upperIndex < num_inst else num_inst
            r.means[step] = np.mean(values[lowerIndex : upperIndex])
            r.variances[step] = np.var(values[lowerIndex : upperIndex])
            r.counts = len(values[lowerIndex : upperIndex])
        r.totalCounts = num_inst

    def __setInput(self, instance, blankPart = 'NONE'):
        for l in xrange(0, Parameters.imageWidth):
            r = l + 1
            for u in xrange(0, Parameters.imageHeight):
                d = u + 1
                num_id = Region.getRegionId(u, d, l, r)
                region = Region.getRegion(num_id)
                if blankPart == 'NONE':
                    region.setBaseValues(instance.getValue(u, l))
                elif blankPart == 'LEFT':
                    if l < Parameters.imageWidth / 2:
                        region.setBaseValuesForBlank()
                    else:
                        region.setBaseValues(instance.getValue(u, l))
                elif blankPart == 'BOTTOM':
                    if u < Parameters.imageHeight / 2:
                        region.setBaseValues(instance.getValue(u, l))
                    else:
                        region.setBaseValuesForBlank()
                    
    def printLearnedModel(self):
        for region_id in self.__pixelRegionId:
            region = Region.getRegion(region_id)
            for sumNode in region.sumNodes:
                print sumNode.getLogValue()
            print region.prodNodes
                
    
    def saveModel(self):
        """
            Save the learned model to a XML file
        """
        pass
    
    def addTrainingSet(self, dataset):
        # dataset is a list of Instance objects
        self.__dataset = dataset

    def getTrainingSetSize(self):
        return len(self.__dataset)

    def getTrainingInstance(self, index):
        assert index < len(self.__dataset)
        return self.__dataset[index]
    
    def initialize(self):
        """
            Initialize each small region of the picture
            and allocate each region a Region object. 
        """
        # for coarse regions : only work with patches larger
        # than 4 x 4 pixels

        # for notation simplicity
        iw = Parameters.imageWidth
        ih = Parameters.imageHeight
        br = Parameters.baseResolution
        for colStepSize in xrange(1, self.__coarseColStep + 1):
            for rowStepSize in xrange(1, self.__coarseRowStep + 1):
                # skip the 4 x 4 pixels patches
                if colStepSize == 1 and rowStepSize == 1:
                    continue
                # find the range of a patch
                # l = left, r = right
                # d = down, u = up
                for l in xrange(0, iw - colStepSize * br + 1, br):
                    r = l + colStepSize * br
                    for u in xrange(0, ih - rowStepSize * br + 1, br):
                        d = u + rowStepSize * br
                        
                        num_id = Region.getRegionId(u, d, l, r)
                        region = Region.getRegion(num_id)
                        self.__coarseRegionId.append(num_id)
                        
                        if colStepSize == self.__coarseColStep \
                           and rowStepSize == self.__coarseRowStep:
                            # this is the root region
                            region.allocateSumNodes(1)
                            self.__rootRegion = region
                            self.__rootSumNode = region.getSumNode(0)
                        else:
                            region.allocateSumNodes(Parameters.numSumNodePerRegion)
        # for fine regions: work with 4 x 4 pixels patches
        for colStepSize in xrange(0, self.__coarseColStep):
            for rowStepSize in xrange(0, self.__coarseRowStep):
                for pixelColStep in xrange(1, br + 1):
                    for pixelRowStep in xrange(1, br + 1):
                        for l in xrange(colStepSize * br, (colStepSize + 1) * br - pixelColStep + 1):
                            r = l + pixelColStep
                            for u in xrange(rowStepSize * br, (rowStepSize + 1) * br - pixelRowStep + 1):
                                d = u + pixelRowStep
                                
                                num_id = Region.getRegionId(u, d, l, r)
                                region = Region.getRegion(num_id)

                                if pixelRowStep == 1 and pixelColStep == 1:
                                    self.__pixelRegionId.append(num_id)
                                else:
                                    self.__fineRegionId.append(num_id)
                                
                                if pixelColStep == 1 and pixelRowStep == 1:
                                    self.__initUnitRegion(region)
                                else:
                                    region.allocateSumNodes(Parameters.numSumNodePerRegion)
        
    def MAPinference(self, index, instance):
        """
            Bottom-up inference
        """       
        self.__setInput(instance)
        # for fine regions: work with 4 x 4 pixels patches
        print '......Inference on fine regions'
        for region_id in self.__fineRegionId:
            region = Region.getRegion(region_id)
            region.MAPinference(index, instance)
        print '......Inference on coarse regions'
        for region_id in self.__coarseRegionId:
            region = Region.getRegion(region_id)
            region.MAPinference(index, instance)
                        
    def setParseToMAP(self, index):
        self.__rootRegion.setParseToMAP(index)
        
    def setParseToMAPFromBuffer(self):
        while Utility.parseBuffer:
            regionRightMax = Utility.parseBuffer.pop()
            regionLeftMax = Utility.parseBuffer.pop()
            regionRightId = Utility.parseBuffer.pop()
            regionLeftId = Utility.parseBuffer.pop()
            mapSumNodeIndex = Utility.parseBuffer.pop()
            region_id = Utility.parseBuffer.pop()
            region = Region.getRegion(region_id)
            region.setParseToMAPFromBuffer(mapSumNodeIndex, \
                                           regionLeftId,
                                           regionRightId,
                                           regionLeftMax,
                                           regionRightMax)
            
    def clearParseToMAP(self, index):
        self.__rootRegion.clearParseToMAP(index)

    def clearParseToMAPFromBuffer(self):
        while Utility.parseBuffer:
            regionRightMax = Utility.parseBuffer.pop()
            regionLeftMax = Utility.parseBuffer.pop()
            regionRightId = Utility.parseBuffer.pop()
            regionLeftId = Utility.parseBuffer.pop()
            mapSumNodeIndex = Utility.parseBuffer.pop()
            region_id = Utility.parseBuffer.pop()
            region = Region.getRegion(region_id)
            region.clearParseToMAPFromBuffer(mapSumNodeIndex, \
                                           regionLeftId, \
                                           regionRightId, \
                                           regionLeftMax, \
                                           regionRightMax)
    
    def clearUnusedDecomp(self):
        """
            Top-down scan each region and clear all the
            unused decomposition in each region.
        """

        # clear coarse region
        for region_id in self.__coarseRegionId:
            region = Region.getRegion(region_id)
            # first of all, gather all the decomposition in use
            # (the decomposition is child of one of the SumNode)
            alive = []
            for node in region.sumNodes:
                if node.getNumOfChildren() > 0:
                    alive.extend(node.getChildren().keys())
            all_decomps = region.prodNodes.keys()
            # find dead decomp using set compliment operation
            dead = set(all_decomps) - set(alive)
            # clear these dead decompositions
            for ddp in dead:
                # remove in local region
                del region.prodNodes[ddp]
                # remove in global decomposition dict
                Decomposition.deleteDecomp(ddp)
        #clear fine region
        for region_id in self.__fineRegionId:
            region = Region.getRegion(region_id)
            # first of all, gather all the decomposition in use
            # (the decomposition is child of one of the SumNode)
            alive = []
            for node in region.sumNodes:
                if node.getNumOfChildren() > 0:
                    alive.extend(node.getChildren().keys())
            all_decomps = region.prodNodes.keys()
            # find dead decomp using set compliment operation
            dead = set(all_decomps) - set(alive)
            # clear these dead decompositions
            for ddp in dead:
                # remove in local region
                del region.prodNodes[ddp]
                # remove in global decomposition dict
                Decomposition.deleteDecomp(ddp)
        

    def getLogLikelihood(self, instance):
        """
            Get the per instance log-likelihood by setting
            the input to the instance and evaluate the SPN,
            the value of the root is the log-likelihood
        """
        self.__setInput(instance)
        self.__evaluateSPN()
        return self.__rootSumNode.getLogValue()

    def __evaluateSPN(self):
        """
            bottom-up evaluate the SPN
        """

        # fine region
        for region_id in self.__fineRegionId:
            region = Region.getRegion(region_id)
            region.evaluate()
        # coarse region
        for region_id in self.__coarseRegionId:
            region = Region.getRegion(region_id)
            region.evaluate()
                        
    def __differentiateSPN(self):
        # all the nodes in the SPN should have initialized
        # the value of their logDerivatives by their super
        # class's contructor, or we should call
        #           self.__initDerivative()

        self.__rootSumNode.setLogDerivative(0.0)
        self.__rootSumNode.passDerivative()

        for decomp_id in self.__rootSumNode.getChildren():
            prodNode = self.__rootSumNode.getChild(decomp_id)
            prodNode.passDerivative()
   
        for region_id in reversed(self.__coarseRegionId[:-1]):
            region = Region.getRegion(region_id)
            region.passDerivative()
        
        for region_id in reversed(self.__fineRegionId):
            region = Region.getRegion(region_id)
            region.passDerivative()

    def __computeMarginal(self, region):
        max_value = 100
        det = 0.0
        numer = 0.0
        for node in region.sumNodes:
            if node.getLogDerivative() == Node.ZERO:
                continue
            if max_value == 100 or node.getLogDerivative() > max_value:
                max_value = node.getLogDerivative()
        for i, node in enumerate(region.sumNodes):
            if node.getLogDerivative() == Node.ZERO:
                continue
            p = np.exp(node.getLogDerivative() - max_value)
            det += p * region.means[i]
            numer += p
        return det / numer
     
    ###########################################################                    
    #                 Image Completion part                   #
    ###########################################################
    def completeLeftImage(self, instance, mode = 'MARGINAL'):
        print 'Complete the left part of the corrupted image'
        if mode == 'MARGINAL':
            self.__completeImageByMarginal(instance, part = 'LEFT')
            
        elif mode == 'MAP':
            self.__completeImageByMAP(instance, part = 'LEFT')

    def __completeImageByMarginal(self, instance, part):
        self.__setInput(instance, part)
        self.__evaluateSPN()
        self.__differentiateSPN()

        
        for l in xrange(0, Parameters.imageWidth):
            r = l + 1
            for u in xrange(0, Parameters.imageHeight):
                d = u + 1
                num_id = Region.getRegionId(u, d, l, r)
                region = Region.getRegion(num_id)
                if part == 'LEFT':
                    if l < Parameters.imageWidth / 2:
                        tmp = self.__computeMarginal(region)
                        Utility.parseBuffer.append( \
                            Utility.getIntValue(instance, tmp))
                    else:
                        Utility.parseBuffer.append( \
                            Utility.getIntValue(instance, instance.getValue(u, l)))
                if part == 'BOTTOM':
                    if u < Parameters.imageHeight / 2:
                        Utility.parseBuffer.append( \
                            Utility.getIntValue(instance, instance.getValue(u, l)))
                    else:
                        tmp = self.__computeMarginal(region)
                        Utility.parseBuffer.append( \
                            Utility.getIntValue(instance, tmp))
    
    def completeBottomImage(self, instance, mode = 'MARGINAL'):
        if mode == 'MARGINAL':
            self.__completeBottomImageByMarginal(instance)
        elif mode == 'MAP':
            self.__completeLeftImageByMAP(instance)

    
        
if __name__ == '__main__':
    s = SPN()
    data = Dataset()
    data.loadData('olivetti.raw')
    
    s.addTrainingSet(data.getTrainingSet())
    print s.getTrainingSetSize()
    s.initialize()
