from Node import Node
import numpy as np
class SumNode(Node):
    """
        Summation Node of the SPN
    """
    def __init__(self):
        """
            each child is characterized by a (decomposition, ProdNode)
            pair, and for each child they have corresponding counts which
            denotes how many times they have won in the inference process.
        """
        Node.__init__(self)
        self.__counts = 0
        self.__children = dict()
        self.__children_counts = dict()
        
    def __str__(self):
        return 'SumNode with value : [%f]' % self.getLogValue()

    def getNumOfChildren(self):
        return len(self.__children)

    def getChildren(self):
        return self.__children
    
    def getChild(self, decomp_id):
        return self.__children[decomp_id]
    
    def addChild(self, decomp_id, node, count):
        if decomp_id not in self.__children:
            self.__children[decomp_id] = node
        if decomp_id not in self.__children_counts:
            self.__children_counts[decomp_id] = count
        else:
            self.__children_counts[decomp_id] = \
                            count + self.getChildCounts(decomp_id)
        self.__counts += count

    def removeChild(self, decomp_id, count):
        cnt = self.getChildCounts(decomp_id)
        cnt -= count
        if cnt == 0:
            del self.__children[decomp_id]
            del self.__children_counts[decomp_id]
        else:
            self.__children_counts[decomp_id] = cnt
        self.__counts -= count
        # This node becomes free
        if self.__counts == 0:
            self.setLogValue(Node.ZERO)
    
    def getCounts(self):
        return self.__counts
    
    def getChildCounts(self, decomp_id):
        assert decomp_id in self.__children_counts
        return self.__children_counts[decomp_id]
    
    def evaluate(self):
        """
            Evaluate the value of the SumNode
            in log scale
        """
        max_decomp = 0
        max_value = 0.0
        v = 0.0
        # find the largest log-value in children
        # for computing the log-exponentials
        for d in self.__children:
            # should be Product Node
            node = self.__children[d]
            value = node.getLogValue()
            if value == Node.ZERO:
                continue
            if max_decomp == 0 or value > max_value:
                max_decomp = d
                max_value = value

        if max_decomp == 0:
            self.setLogValue(Node.ZERO)

        for d in self.__children:
            if d not in self.__children_counts:
                continue
            cnts = self.__children_counts[d]
            node = self.__children[d]
            value = node.getLogValue()
            if value == Node.ZERO:
                continue
            v += cnts * np.exp(value - max_value)
        self.setLogValue(np.log(v / self.__counts) + max_value)
            
            
    def passDerivative(self):
        """
            Pass the derivative to its children
        """
        # this will happen when this SumNode is unused
        if self.getLogDerivative() == Node.ZERO:
            return 
        for decomp_id in self.getChildren():
            prodNode = self.getChild(decomp_id)
            tmp = self.getLogDerivative() \
                  + np.log(float(self.getChildCounts(decomp_id)) / self.getCounts())

            if prodNode.getLogDerivative() == Node.ZERO:
                prodNode.setLogDerivative(tmp)
            else:
                prodNode.setLogDerivative( \
                    np.logaddexp(tmp, self.getLogDerivative()))
                

if __name__ == '__main__':
    a = SumNode()
    b = SumNode()

    print isinstance(a, int)

