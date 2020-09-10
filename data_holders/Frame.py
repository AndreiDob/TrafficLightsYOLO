from config import NETWORK_IMAGE_SIZE
from utils.cv_utils import load_resized_normalized_image, show_one_image


class Frame:
    """
        Class holding properties of a ground truth image.

        Attributes:
            pathToImage(string):          the path to the location of the image
            regions(list<Region>):        the ground truth boxes contained in the image
            loadedImage(ndarray):         the image loaded as a numpy array
        """

    def __init__(self, pathToImage, regions, loadedImage=None):
        self.pathToImage = pathToImage
        self.regions = regions
        if loadedImage is None:
            self.loadedImage = load_resized_normalized_image(pathToImage)
        else:
            self.loadedImage = loadedImage
