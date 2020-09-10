from config import *


class Region:
    """
        Class holding properties of a ground truth labeled box.

        Attributes:
            centerX(float):          X coordinate of center of the box in percents of image width
            centerY(float):          Y coordinate of center of the box in percents of image height
            width(float):              Width of bounding box label
            height(float):             Height of bounding box label

            topLeftX(float):              X coordinate of upper left corner of bounding box label
            topLeftY(float):              Y coordinate of upper left corner of bounding box label
            bottomRightX(float):          X coordinate of bottom right corner of bounding box label
            bottomRightY(float):          Y coordinate of bottom right corner of bounding box label

            grid10Row(int):      The row of the cell responsible for predicting the box in the big prediction scale
            grid10Column(int):   The column of the cell responsible for predicting the box in the big prediction scale

            grid20Row(int):      The row of the cell responsible for predicting the box in the middle prediction scale
            grid20Column(int):   The column of the cell responsible for predicting the box in the middle prediction scale

            grid40Row(int):      The row of the cell responsible for predicting the box in the small prediction scale
            grid40Column(int):   The column of the cell responsible for predicting the box in the small prediction scale
        """

    def __init__(self, cls, centerX, centerY, width, height):
        self.cls = cls

        self.width = width
        self.height = height
        self.centerX = centerX
        self.centerY = centerY

        self.topLeftX = centerX - (width / 2)
        self.topLeftY = centerY - (height / 2)
        self.bottomRightX = centerX + (width / 2)
        self.bottomRightY = centerY + (height / 2)

        self.grid10Row = int(self.centerY * GRID_SIZE_SCALE_1[1]) - int(self.centerY >= 1.)
        self.grid10Column = int(self.centerX * GRID_SIZE_SCALE_1[0]) - int(self.centerX >= 1.)

        self.grid20Row = int(self.centerY * GRID_SIZE_SCALE_2[1]) - int(self.centerY >= 1.)
        self.grid20Column = int(self.centerX * GRID_SIZE_SCALE_2[0]) - int(self.centerX >= 1.)

        self.grid40Row = int(self.centerY * GRID_SIZE_SCALE_3[1]) - int(self.centerY >= 1.)
        self.grid40Column = int(self.centerX * GRID_SIZE_SCALE_3[0]) - int(self.centerX >= 1.)

    def getCellPositionByGridSize(self, gridSize):
        """
                Method for getting the coordinates of the cell responsible for predicting the box by scale
                at the correct position

                Args:
                    gridSize:int - the scale for which to get the row and column
                Returns:
                    tuple(row, column)

                """
        if gridSize == 10:
            return self.grid10Row, self.grid10Column
        elif gridSize == 20:
            return self.grid20Row, self.grid20Column
        elif gridSize == 40:
            return self.grid40Row, self.grid40Column
        else:
            raise Exception("Wrong grid size")
