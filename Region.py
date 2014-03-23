from Parameters import Parameters
from SumNode import SumNode
from ProductNode import ProductNode
from Node import Node
from Decomposition import Decomposition
from Utility import Utility
import numpy as np
class Region():
    """
        Using the coordinates of the four corners
        as the identifier for a specific region of
        the picture.
    """
    __region_dict = {}

    def __init__(self, id_num, ru, rd, cl, cr):
        self.id = id_num
        self.rowUp = ru
        self.rowDown = rd
        self.columnLeft = cl
        self.columnRight = cr
        # the size statistics of the region
        self.width = rd - ru
        self.height = cr - cl
        assert self.width <= Parameters.imageWidth
        assert self.height <= Parameters.imageHeight
        if self.width <= Parameters.baseResolution \
           and self.height <= Parameters.baseResolution:
            self.interval = 1
        else:
            self.interval = 4
        # SumNodes for the region
        self.sumNodes = []
        self.mapSumNodeIndex = -1
        self.mapSumNodeProb = 100
        self.mapProdNodeProb = 100
        # dict for finding best decomposition
        # each decomposition corresponds to a ProdNode
        self.prodNodes = dict() # key - decomp_id,  value - ProdNode
        self.decompPerInstance = dict() # key - instance index,  value - decomp_id
        self.mapSumNodePerInstance = dict() # key - instance index, value - mapSumNode index
        # tag the best decomposition of each SumNode of this region
        self.bestDecompPerSumNode = []

    def __str__(self):
        return 'Region : <%d,%d,%d,%d>' % \
              (self.columnLeft, self.columnRight, self.rowUp, self.rowDown)
    
    def setBaseValues(self, value):
        """
            This will only be called in unit regions
        """
        assert len(self.sumNodes) == Parameters.numSumNodePerPixel
        max_val = 0.0
        self.mapSumNodeIndex = -1
        for i, node in enumerate(self.sumNodes):
            tmp = self.GaussianKernel(value, self.means[i])
            node.setLogValue(tmp)
            if self.mapSumNodeIndex == -1 or tmp > max_val:
                self.mapSumNodeIndex = i
                max_val = tmp
    def setBaseValuesForBlank(self):
        """
            This will only be called in unit regions
        """
        assert len(self.sumNodes) == Parameters.numSumNodePerPixel
        self.mapSumNodeIndex = -1
        for node in self.sumNodes:
            node.setLogValue(0.0)
        
    def GaussianKernel(self, value, mean):
        return -((value - mean) ** 2) / 2
    
    def allocateSumNodes(self, num):
        self.sumNodes = [SumNode()] * num

    def getSumNode(self, index):
        assert len(self.sumNodes) > 0
        return self.sumNodes[index]

    def MAPinference(self, index, instance):
        
        self.mapSumNodeIndex = -1
        self.mapSumNodeProb = 100
        self.mapProdNodeProb = 100

        # randomly choose a unused SumNode
        unusedNodes = []
        for i, node in enumerate(self.sumNodes):
            if node.getNumOfChildren() == 0:
                unusedNodes.append(i)
        nodeIndex = -1
        if len(unusedNodes) > 0:
            nodeIndex = np.random.randint(0, len(unusedNodes))
            nodeIndex = unusedNodes[nodeIndex]
        # try to find a better decomposition of this region
        cl = self.columnLeft
        cr = self.columnRight
        ru = self.rowUp
        rd = self.rowDown
        step = self.interval
        decompOptions = []
        # try to decompose into left and right parts
        for index in xrange(cl + step, cr, step):
            lr_id = Region.getRegionId(ru, rd, cl, index)
            rr_id = Region.getRegionId(ru, rd, index, cr)
            lr = Region.getRegion(lr_id)
            rr = Region.getRegion(rr_id)
            snl = lr.sumNodes[lr.mapSumNodeIndex]
            snr = rr.sumNodes[rr.mapSumNodeIndex]
            
            max_value = 0.0
            if snl.getLogValue() == Node.ZERO \
               or snr.getLogValue() == Node.ZERO:
                max_value = Node.ZERO
            else:
                max_value = snl.getLogValue() + snr.getLogValue()
            if len(decompOptions) == 0 \
               or max_value > self.mapProdNodeProb:
                self.mapProdNodeProb = max_value
                decompOptions = []

            if max_value == self.mapProdNodeProb:
                str_id = Decomposition.getDecompId( \
                    lr_id, rr_id, lr.mapSumNodeIndex, rr.mapSumNodeIndex)
                decompOptions.append(str_id)
        # try to decompose into up and down parts
        for index in xrange(ru + step, rd, step):
            ur_id = Region.getRegionId(ru, index, cl, cr)
            dr_id = Region.getRegionId(index, rd, cl, cr)
            ur = Region.getRegion(ur_id)
            dr = Region.getRegion(dr_id)
            snu = ur.sumNodes[ur.mapSumNodeIndex]
            snd = dr.sumNodes[dr.mapSumNodeIndex]
            max_value = 0.0
            if snu.getLogValue() == Node.ZERO \
               or snd.getLogValue() == Node.ZERO:
                max_value = Node.ZERO
            else:
                max_value = snu.getLogValue() + snd.getLogValue()
 
            if len(decompOptions) == 0 \
               or max_value > self.mapProdNodeProb:
                self.mapProdNodeProb = max_value
                decompOptions = []

            if max_value == self.mapProdNodeProb:
                str_id = Decomposition.getDecompId( \
                    ur_id, dr_id, ur.mapSumNodeIndex, dr.mapSumNodeIndex)
                decompOptions.append(str_id)

        # randomly choose a decomposition
        idx = np.random.randint(0, len(decompOptions))
        mapDecomp = decompOptions[idx]
        
        # evaluate existing ProdNode/Decomp on this instance
        for d in self.prodNodes:
            node = self.prodNodes[d]
            node.evaluate()

            
        # temperary list for finding maxSumNodeIndex
        mapSumNodeOptions = []
        bestDecompOptions = []
        self.bestDecompPerSumNode = [''] * len(self.sumNodes)

        
        for i, node in enumerate(self.sumNodes):
            if node.getNumOfChildren() == 0:
                continue
            node.evaluate()
            
            mapSumNodeProbOption = 0
            
            for decomp_id in node.getChildren():
                child = node.getChild(decomp_id)
                # the following two equations will calculate
                # the new value if we vote this child in the
                # inference process.
                
                old_value = node.getLogValue() \
                            + np.log(node.getCounts())
                child_value = child.getLogValue()
                value = 0.0

                # using the Log-Exponential trick to calculate
                # log(exp(.) + exp(.)) for avoiding underlow/overflow
##                if old_value > child_value:
##                    value = old_value + np.log(1 + np.exp(child_value - old_value))
##                else:
##                    value = child_value + np.log(1 + np.exp(old_value - child_value))
                value = np.logaddexp(old_value, child_value)
                if len(bestDecompOptions) == 0 \
                   or value > mapSumNodeProbOption:
                    bestDecompOptions = []
                    mapSumNodeProbOption = value
                if value == mapSumNodeProbOption:
                    bestDecompOptions.append(decomp_id)
            # the is a new Decomposition (child)
            if mapDecomp not in node.getChildren():
                
                value = self.mapProdNodeProb
                
                # this new child is not the only effective child
                if node.getLogValue() != Node.ZERO:
                    value = node.getLogValue() + np.log(node.getCounts())
                    # same log exponential trick
##                    if self.mapProdNodeProb > value:
##                        value = self.mapProdNodeProb \
##                                + np.log(1 + np.exp(value - self.mapProdNodeProb))
##                    else:
##                        value = value + \
##                                np.log(1 + np.exp(self.mapProdNodeProb - value))
                    value = np.logaddexp(self.mapProdNodeProb, value)
                value -= Parameters.prior
                if len(bestDecompOptions) == 0 \
                   or value > mapSumNodeProbOption:
                    bestDecompOptions = []
                    mapSumNodeProbOption = value
                    bestDecompOptions.append(mapDecomp)
            # get the final log value of this SumNode
            node.setLogValue(mapSumNodeProbOption \
                             - np.log(node.getCounts() + 1))
            # find the new best decomposition
            # (maybe one of the old one or the new one)
            length = len(bestDecompOptions)
            index = np.random.randint(0, length)
            self.bestDecompPerSumNode[i] = bestDecompOptions[index]

            if len(mapSumNodeOptions) == 0 \
               or node.getLogValue() > self.mapSumNodeProb:
                self.mapSumNodeProb = node.getLogValue()
                mapSumNodeOptions = []
            if node.getLogValue() == self.mapSumNodeProb:
                mapSumNodeOptions.append(i)
        # find the map SumNode
        if nodeIndex >= 0:
            node = self.sumNodes[nodeIndex]
            node.setLogValue(self.mapProdNodeProb - \
                             np.log(node.getCounts() + 1) - \
                             Parameters.prior)
            self.bestDecompPerSumNode[nodeIndex] = mapDecomp
            if len(mapSumNodeOptions) == 0 \
               or node.getLogValue() > self.mapSumNodeProb:
                self.mapSumNodeProb = node.getLogValue()
                mapSumNodeOptions = []
                mapSumNodeOptions.append(nodeIndex)
                
        length = len(mapSumNodeOptions)
        index = np.random.randint(0, length)
        self.mapSumNodeIndex = mapSumNodeOptions[index]
        
    def setParseToMAP(self, index):
        # skip unit region
        if self.width == 1 and self.height == 1:
            return

        if len(self.sumNodes) == 1:
            self.mapSumNodePerInstance[index] = 0
            
        mapSumNodeIndex = self.mapSumNodePerInstance[index]
        decomp_id = self.bestDecompPerSumNode[mapSumNodeIndex]
        self.decompPerInstance[index] = decomp_id

        decomp = Decomposition.getDecomp(decomp_id)

        regionLeft = Region.getRegion(decomp.regionLeftId)
        regionRight = Region.getRegion(decomp.regionRightId)

        regionLeft.mapSumNodePerInstance[index] = decomp.regionLeftMax
        regionRight.mapSumNodePerInstance[index] = decomp.regionRightMax

        # we are working in single machine, so record updates anyway
        Utility.parseBuffer.append(self.id)      
        Utility.parseBuffer.append(mapSumNodeIndex)       
        Utility.parseBuffer.append(decomp.regionLeftId)     
        Utility.parseBuffer.append(decomp.regionRightId)
        Utility.parseBuffer.append(decomp.regionLeftMax)
        Utility.parseBuffer.append(decomp.regionRightMax)
        

        # check whether or not the ProdNode for the decomposition is created
        if decomp_id not in self.prodNodes:
            node = ProductNode()
            self.prodNodes[decomp_id] = node
            leftChild = regionLeft.getSumNode(decomp.regionLeftMax)
            rightChild = regionRight.getSumNode(decomp.regionRightMax)
            node.addChild(leftChild)
            node.addChild(rightChild)
            
        # recursively parse the tree
        regionLeft.setParseToMAP(index)
        regionRight.setParseToMAP(index)

    def setParseToMAPFromBuffer(self, \
                                maxSumNodeIndex, \
                                regionLeftId, \
                                regionRightId, \
                                regionLeftMax, \
                                regionRightMax):
        if self.width == 1 and self.height == 1:
            return
        decomp_id = Decomposition.getDecompId(regionLeftId, \
                                           regionRightId, \
                                           regionLeftMax, \
                                           regionRightMax)
        
        sumNode = self.sumNodes[maxSumNodeIndex]
        prodNode = self.prodNodes[decomp_id]
        sumNode.addChild(decomp_id, prodNode, 1)

    def clearParseToMAP(self, index):
        if index not in self.mapSumNodePerInstance:
            return
        if self.width == 1 and self.height == 1:
            return
        mapSumNodeIndex = self.mapSumNodePerInstance[index]
        decomp_id = self.decompPerInstance[index]

        del self.mapSumNodePerInstance[index]
        del self.decompPerInstance[index]

        decomp = Decomposition.getDecomp(decomp_id)
        regionLeft = Region.getRegion(decomp.regionLeftId)
        regionRight = Region.getRegion(decomp.regionRightId)
        #record changes
        Utility.parseBuffer.append(self.id)
        Utility.parseBuffer.append(mapSumNodeIndex)
        Utility.parseBuffer.append(decomp.regionLeftId)     
        Utility.parseBuffer.append(decomp.regionRightId)
        Utility.parseBuffer.append(decomp.regionLeftMax)
        Utility.parseBuffer.append(decomp.regionRightMax)

        # recursively parse the tree
        regionLeft.clearParseToMAP(index)
        regionRight.clearParseToMAP(index)

    def clearParseToMAPFromBuffer(self, \
                                mapSumNodeIndex, \
                                regionLeftId, \
                                regionRightId, \
                                regionLeftMax, \
                                regionRightMax):
        if self.width == 1 and self.height == 1:
            return
        decomp_id = Decomposition.getDecompId(regionLeftId, \
                                           regionRightId, \
                                           regionLeftMax, \
                                           regionRightMax)
        sumNode = self.sumNodes[mapSumNodeIndex]
        sumNode.removeChild(decomp_id, 1)


    def evaluate(self):
        """
            Evaluate all the SumNodes and ProdNodes in this region
        """
        for prodNode in self.prodNodes.values():
            prodNode.evaluate()
        for node in self.sumNodes:
            if node.getNumOfChildren() > 0:
                node.evaluate()
            else:
                node.setLogValue(Node.ZERO)
    def passDerivative(self):
        """
            Top-down derivative propagation
        """
        for sumNode in self.sumNodes:
            if sumNode.getNumOfChildren() > 0:
                sumNode.passDerivative()
    
        for prodNode in self.prodNodes.values():
            prodNode.passDerivative()
                    
    @staticmethod
    def getRegionId(rowUp, rowDown, columnLeft, columnRight):
        id_num = ((Parameters.imageWidth * rowUp + rowDown - 1) \
                 * Parameters.imageHeight + columnLeft) \
                 * Parameters.imageHeight + columnRight - 1
        if id_num not in Region.__region_dict:
            Region.__region_dict[id_num] = \
                        Region(id_num, rowUp, rowDown, columnLeft, columnRight)
        return id_num

    @staticmethod
    def getRegion(id_num):
        return Region.__region_dict[id_num]
    
if __name__ == '__main__':
    r = Region.getRegionId(0,1,1,64)
    ri = Region.getRegion(r)
    print ri.id
    a = [SumNode()] * 20
    print a
