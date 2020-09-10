import random
from math import log

from scipy.special._ufuncs import logit
from tensorflow.python.keras.utils import Sequence
import numpy as np

from data_holders.Frame import Frame
from config import NETWORK_IMAGE_SIZE, GRID_SIZE_SCALE_1, GRID_SIZE_SCALE_2, GRID_SIZE_SCALE_3
from utils.cv_utils import show_bboxes_and_image_centers
from utils.data_augmentation_utils import get_flipped_frames, get_exposure_augmented_frames, \
    get_crooped_frame, get_zoomed_out_frame, get_gaussian_noised_frame
from utils.network_utils import getCellDimensionsByGridSize


class BatchGenerator(Sequence):
    """
            Class used to feed the network both images(X) as well as ground truths (Y) during training.

            Attributes:
                frames(list<Frame>):      frames from which to extract and process the data required for training
                anchors(list<int>):       anchors to be used when crating the train data
                batch_size(int):          number of examples on which to run feed-forward steps,
                                          then run a back-propagation step using the mean of the losses
                useAugmentation(bool):    whether to augment the train images
                showImages(bool):         whether to display the train images
            """

    def __init__(self,
                 frames,
                 anchors,
                 batch_size=1,
                 useAugmentation=False,
                 showImages=False,
                 ):
        self.frames = frames
        self.anchors = anchors
        self.batch_size = batch_size  # // 8
        self.useAugmentation = useAugmentation
        self.showImages = showImages

    # batches per epoch
    def __len__(self):
        return int(np.ceil(float(len(self.frames)) / self.batch_size))

    def __getitem__(self, idx):
        """
            Method called by Keras when it needs a new batch of examples to run on during training

            Args:
                idx: int - batch id
            Returns:
                    networkY:list<ndarray> - the processed labels used by the network as ground truths
                    networkX:ndarray - the processed images used by the network as image input when training
            """
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.frames):
            r_bound = len(self.frames)
            l_bound = r_bound - self.batch_size

        framesToBeProcessed = []
        # copying frames to new list in order to force loading of image in each frame object
        for frame in self.frames[l_bound:r_bound]:
            framesToBeProcessed.append(Frame(frame.pathToImage, frame.regions))

        augmentedFrames = []
        for frame in framesToBeProcessed:
            augmentedFrames.append(self.get_augmented_frame(frame))

        networkX, networkY = self.create_train_data(augmentedFrames, self.anchors, showImages=self.showImages)

        return networkX, networkY

    def get_augmented_frame(self, frame):
        """
            Method for mapping a DiveuImage to a Frame for further processing.

            Args:
                image: DriveuImage - images to map
            Returns:
                new Frame class containing the DriveuImage coordinates

            """
        if not self.useAugmentation:
            return frame

        if bool(random.getrandbits(1)):
            frame = get_gaussian_noised_frame(frame, random.uniform(0, 0.0008))

        if bool(random.getrandbits(1)):
            frame = get_exposure_augmented_frames([frame])[0]

        if bool(random.getrandbits(1)):
            frame = get_flipped_frames([frame])[0]

        # if bool(random.getrandbits(1)):
        #     # print("translation")
        #     frame = get_list_of_translations(frame, translationCodes=[random.randint(1, 4)])[0]

        zoomCmd = random.randint(1, 3)
        if zoomCmd == 1:
            zoomFactor = random.uniform(0.0, 0.3)
            offsetX = random.uniform(-zoomFactor, zoomFactor)
            offsetY = random.uniform(-zoomFactor, 0)
            frame = get_crooped_frame(frame,  # percents of image size
                                      leftX=zoomFactor + offsetX,
                                      rightX=1 - zoomFactor + offsetX,
                                      topY=zoomFactor + offsetY,
                                      bottomY=1 - zoomFactor + offsetY
                                      )
        elif zoomCmd == 2:
            zoomFactor = random.uniform(0.7, 1)
            frame = get_zoomed_out_frame(frame, int(zoomFactor * NETWORK_IMAGE_SIZE[0]),
                                         int(zoomFactor * NETWORK_IMAGE_SIZE[1]))

        return frame

    def create_train_data(self, frames, anchors, showImages=False):
        '''
                Method for creating the network input from labeled frames and anchors.

                Args:
                    frames: list<Frame> - frames from which to get data
                    anchors:list<int> - anchors to be used when calculating data for network
                Returns:
                    networkY:list<ndarray> - the processed labels used by the network as ground truths
                    networkX:ndarray - the processed images used by the network as image input when training

                '''
        # rows(Y) first, columns (X) last
        networkY = [np.zeros((len(frames), GRID_SIZE_SCALE_1[1], GRID_SIZE_SCALE_1[0], 24)),
                    np.zeros((len(frames), GRID_SIZE_SCALE_2[1], GRID_SIZE_SCALE_2[0], 24)),
                    np.zeros((len(frames), GRID_SIZE_SCALE_3[1], GRID_SIZE_SCALE_3[0], 24))]
        networkX = np.zeros((len(frames), NETWORK_IMAGE_SIZE[1], NETWORK_IMAGE_SIZE[0], 3))
        frameCounter = -1
        for frame in frames:
            frameCounter += 1
            for region in frame.regions:
                gridSize, anchorPosition, bestAnchor = self.get_anchor_for_region(region, anchors)
                # get the adequate cell of a region by grid size
                regionRow, regionColumn = region.getCellPositionByGridSize(gridSize)
                positionByGridSize = int(log(gridSize // 10, 2))
                cell_prediction = networkY[positionByGridSize][frameCounter][regionRow][regionColumn]
                # begining position of area to be written
                offset = anchorPosition * 8

                tx, ty, tw, th = self.calculate_network_bbox_parameters(region, bestAnchor, gridSize)
                confidence = 1

                cell_prediction[offset] = tx
                cell_prediction[offset + 1] = ty
                cell_prediction[offset + 2] = tw
                cell_prediction[offset + 3] = th
                cell_prediction[offset + 4] = confidence

                if region.cls == "red":
                    cell_prediction[offset + 5] = 1
                elif region.cls == "yellow":
                    cell_prediction[offset + 6] = 1
                elif region.cls == "green":
                    cell_prediction[offset + 7] = 1

                networkY[positionByGridSize][frameCounter][regionRow][regionColumn] = cell_prediction

            if showImages: show_bboxes_and_image_centers(frame.loadedImage, NETWORK_IMAGE_SIZE, frame.regions,
                                                         showFromLoadedImage=True)

            networkX[frameCounter] = frame.loadedImage
        return networkX, networkY

    def calculate_network_bbox_parameters(self, region, bestAnchor, gridSize):
        """
            Method for calculating the parameters required by the network from labeled box data, anchors and prediction scale

            Args:
                region:Region - labeled box
                bestAnchor:tuple(int,int) - best anchor for that region
                gridSize:int - prediction scale that should predict the box
            Returns:
                tx:float - the box center x coordinate used by the network
                ty:float - the box center y coordinate used by the network
                tw:float - the box width coordinate used by the network
                th:float - the box height coordinate used by the network

            """
        # get the adequate cell of a region by grid size
        regionRow, regionColumn = region.getCellPositionByGridSize(gridSize)
        # network cell width and height in pixels
        cellWidth, cellHeight = getCellDimensionsByGridSize(gridSize)

        tx = logit(region.centerX * NETWORK_IMAGE_SIZE[0] / cellWidth - regionColumn)
        ty = logit(region.centerY * NETWORK_IMAGE_SIZE[1] / cellHeight - regionRow)
        tw = np.log(region.width * NETWORK_IMAGE_SIZE[0] / bestAnchor[0])
        th = np.log(region.height * NETWORK_IMAGE_SIZE[1] / bestAnchor[1])

        tx = np.clip(tx, logit(1e-15), logit(1 - 1e-15))
        ty = np.clip(ty, logit(1e-15), logit(1 - 1e-15))
        tw = np.clip(tw, np.log(1e-15), None)
        th = np.clip(th, np.log(1e-15), None)
        return tx, ty, tw, th

    def iou(self, regionWidth, regionHeight, anchorWidth, anchorHeight):
        """
            Method for calculating the IOU between a region and an anchor

            Args:
                regionWidth:float - region width
                regionHeight:float - region height
                anchorWidth:float - anchor width
                anchorHeight:float - anchor height
            Returns:
                int - IOU of the anchor and the region

            """
        widths = [regionWidth, anchorWidth]
        heights = [regionHeight, anchorHeight]
        intersection = min(widths) * min(heights)
        union = regionWidth * regionHeight + anchorWidth * anchorHeight - intersection
        return union / intersection

    def get_anchor_for_region(self, region, anchors):
        """
            Method for finding the best fitting anchor for a certain region

            Args:
                region:Region - region for which to find the best anchor
                anchors:list<int> - anchors to select from
            Returns:
                gridSize:int - best scale for predicting the given region
                anchorPosition:int - each cell in the gris uses 3 anchors. This represents on which of the
                                     three anchor positions is the best anchor situated
                bestAnchor:int - best anchor for the given region

            """
        score = []
        for anchor in anchors:
            score.append(
                self.iou(region.width * NETWORK_IMAGE_SIZE[0], region.height * NETWORK_IMAGE_SIZE[1], anchor[0],
                         anchor[1]))
        gridSize = (2 ** (score.index(min(score)) // 3)) * 10
        anchorPosition = score.index(min(score)) % 3
        bestAnchor = anchors[score.index(min(score))]
        return gridSize, anchorPosition, bestAnchor
