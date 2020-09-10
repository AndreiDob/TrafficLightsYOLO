class BoundBox:
    """
        Class holding properties of a detected object.

        Attributes:
            xmin(int):          X coordinate of upper left corner of bounding box label
            ymin(int):          Y coordinate of upper left corner of bounding box label
            xmax(int):          X coordinate of bottom right corner of bounding box label
            ymax(int):          Y coordinate of bottom right corner of bounding box label
            objness(float):     confidence that the predicted box actually contains a valid object
            classes(list<float>):   3 class probabilities
            label(int)          0,1,2 - the decided class after the post-processing
            score(int)          final confidence in percents
            actual_class(string)    name of the decided class
        """

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        self.actual_class = "???"

    def __str__(self) -> str:
        return str(self.classes)