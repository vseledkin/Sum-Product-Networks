
class Decomposition:
    """
        Represent a decomposition of a region, which
        is useful for remembering decomposed region id
        for subsequent utilization.
    """
    # key --- decomposition id str
    # value --- Decomposition object

    __decomp_dict = {}

    def __init__(self, str_id, l, r, maxl, maxr):
        """
            l --- the id of decomposed region left/up
            r --- the id or decomposed region right/down
            maxl --- the index of the max SumNode of left/up region
            maxr --- the index of the max SumNode of right/down region
        """
        self.id = str_id
        self.regionLeftId = l
        self.regionRightId = r
        self.regionLeftMax = maxl
        self.regionRightMax = maxr

    @staticmethod
    def getDecomp(l, r, maxl, maxr):
        str_id = self.getDecompId(l, r, maxl, maxr)
        if str_id in Decomposition.__decomp_dict:
            d = Decomposition.__decomp_dict[str_id]
    @staticmethod
    def getDecomp(str_id):
        if str_id in Decomposition.__decomp_dict:
            d = Decomposition.__decomp_dict[str_id]
        else:
            #print str_id
            tmp = map(int, str_id.split(' '))
            d = Decomposition(str_id, tmp[0], tmp[1], tmp[2], tmp[3])
            #Decomposition.__decomp_dict[str_id] = d
        return d

    @staticmethod
    def getDecompId(l, r, maxl, maxr):
        str_id = ' '.join(map(str, [l, r, maxl, maxr]))
        return str_id

    @staticmethod
    def deleteDecomp(decomp_id):
        if decomp_id in Decomposition.__decomp_dict:
            del Decomposition.__decomp_dict[decomp_id]
        else:
            return
