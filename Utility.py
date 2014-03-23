class Utility:
    parseBuffer = []
    testSet = []

    @staticmethod
    def getIntValue(instance, value):
        return int(value * instance.getStd() + instance.getMean())

