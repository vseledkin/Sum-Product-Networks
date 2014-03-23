from Node import Node
import numpy as np
class ProductNode(Node):
    """
        Product Node of the SPN
    """
    def __init__(self):
        Node.__init__(self)
        self.__children = []
        
    def __str__(self):
        return 'ProductNode with value : [%f]' % self.getLogValue()

    def addChild(self, node):
        self.__children.append(node)
    
    def evaluate(self):
        """
            Evaluate the value of the ProductNode
            in log scale
        """
        tmp = 0.0
        for node in self.__children:
            value = node.getLogValue()
            if value == Node.ZERO:
                self.setLogValue(Node.ZERO)
                return
            tmp += value
        self.setLogValue(tmp)
    def passDerivative(self):
        """
            Pass the derivative to its children
        """
        # because of unused parent
        if self.getLogDerivative() == Node.ZERO:
            return 
        
        assert self.getLogValue() != Node.ZERO
  
        for sumNode in self.__children:
            tmp = self.getLogDerivative() + \
                  self.getLogValue() - sumNode.getLogValue()
            if sumNode.getLogDerivative() == Node.ZERO:
                sumNode.setLogDerivative(tmp)
            else:
                sumNode.setLogDerivative( \
                    np.logaddexp(sumNode.getLogDerivative(), tmp))
        
