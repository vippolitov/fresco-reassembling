class ColorDescriptor:
    """
    h - histogram
    b - center of mass (colors)
    var - variances of colors before quantization (len(var) = len(h))
    """
    def __init__(self, h, b, var):
        self.h = h
        self.b = b
        self.var = var