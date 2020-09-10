from __future__ import print_function

import pickle

from scipy.special import logit
from sklearn.cluster import KMeans

from data_holders.Frame import Frame
from data_holders.Region import Region
from config import NETWORK_IMAGE_SIZE, GRID_SIZE_SCALE_1, GRID_SIZE_SCALE_2, GRID_SIZE_SCALE_3
from preprocessing.dtld_parsing.driveu_dataset import DriveuDatabase
import matplotlib.pyplot as plt

import argparse
import logging
import sys

import numpy as np

import cv2
from math import log
from utils.network_utils import getCellDimensionsByGridSize
from utils.os_utils import create_dir

np.set_printoptions(suppress=True)

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: "
           "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_width", type=int, default="10", help='Minimum width of the bounding boxes')
    parser.add_argument("--min_height", type=int, default="20", help='Minimum height of the bounding boxes')

    parser.add_argument("--label_file", default="preprocessing/labels/Fulda_all.yml",
                        help='Location of label files to be used')
    parser.add_argument("--image_file", default="driveu_original_images",
                        help='Location of the folder where all the image folders are contained')
    parser.add_argument("--output_file", default="network_files",
                        help='Location where to store the processed annotations')
    return parser.parse_args()


def get_filtered_images(images, width, height):
    """
        Method for filtering DriveuImages by minimum width and height,
        also only with lit traffic lights that are facing the car

        Args:
            images: list<DriveuImage> - images to filter
            width:int - minimum filtering width in pixels
            height:int - minimum filtering height in pixels
        Returns:
            filteredImages:list<DriveuImage> - list of containing the filtered images

        """
    print("Filtering images . . .")
    filteredImages = []
    for idx_d, image in enumerate(images):

        if len(filteredImages) != 0 and len(filteredImages) % 100 == 0:
            print("Images filtered:", len(filteredImages))

        imageFound, _ = image.get_image()
        if imageFound:
            filteredBoxes = []
            for box in image.objects:
                if box.width < width and box.height < height: continue
                # if the lights are not facing the car
                if str(box.class_id)[0] != "1": continue
                # if the traffic light is unlit
                if str(box.class_id)[-2] == "0": continue

                filteredBoxes.append(box)

            if len(filteredBoxes) > 0:
                image.objects = filteredBoxes
                filteredImages.append(image)
    print("Images filtered.")
    return filteredImages


def map_DriveuImage_to_Frame(image):
    """
    Method for mapping a DiveuImage to a Frame for further processing.

    Args:
        image: DriveuImage - images to map
    Returns:
        new Frame class containing the DriveuImage coordinates

    """
    _, img = image.get_image()
    imageHeight = img.shape[0]
    imageWidth = img.shape[1]
    regions = []
    for box in image.objects:
        regionSimple = Region(get_box_class_from_driveu_id(box.class_id),
                              (box.x + box.width / 2) / imageWidth,
                              (box.y + box.height / 2) / imageHeight,
                              box.width / imageWidth,
                              box.height / imageHeight)
        if regionSimple.centerX < 0 or regionSimple.centerX > 1: continue
        if regionSimple.centerY < 0 or regionSimple.centerY > 1: continue
        if regionSimple.width < 0 or regionSimple.width > 1: continue
        if regionSimple.height < 0 or regionSimple.height > 1: continue
        regions.append(regionSimple)
    if len(regions) > 0:
        city = image.file_path.split("//")[-1].split("/")[0]
        name = image.file_path.split("/")[-1].split(".")[0]
        path = 'images/' + city + "/" + name + ".jpg"
        return Frame(path, regions, loadedImage=np.array([]))


def get_box_class_from_driveu_id(id):
    """
    Method for extracting the traffic light state from the Driveu class encoding

    Args:
        id:int - Driveu encoding
    Returns:
        string containing the state

    """
    if str(id)[-2] == "1" or str(id)[-2] == "3":
        return "red"
    elif str(id)[-2] == "2":
        return "yellow"
    elif str(id)[-2] == "4":
        return "green"
    else:
        raise RuntimeError("Something failed in the filtering. No unlit traffic lights should get here.")


def save_converted_images(images):
    '''
    Method for saving DriveuImages images.

    Args:
        images: list<DriveuImage> - images to save

    '''
    print("Saving images . . .")
    for idx_d, image in enumerate(images):
        if idx_d != 0 and idx_d % 100 == 0:
            print("Images saved:", idx_d)
        _, convertedImage = image.get_image()
        city = image.file_path.split("//")[-1].split("/")[0]
        name = image.file_path.split("/")[-1].split(".")[0]
        create_dir('images/' + city)
        savePath = 'images/' + city + "/" + name + ".jpg"
        cv2.imwrite(savePath, convertedImage)
    print("Images saved.")


def k_means_anchor_boxes(frames):
    '''
        Method for finding and displaying the anchors for the frames used.

        Args:
            frames:list<Frame> - the frames from which to get the width and height for calculating the best anchors
        Returns:
            anchors:list<int> - the best anchors for the given frames

        '''
    bbox_sizes = []

    for frame in frames:
        for region in frame.regions:
            bbox_sizes.append([region.width * NETWORK_IMAGE_SIZE[0], region.height * NETWORK_IMAGE_SIZE[1]])

    x = [int(box[0]) for box in bbox_sizes]
    y = [int(box[1]) for box in bbox_sizes]

    plt.scatter(x, y, c="#0000FF")

    # convert it to numpy array
    bbox_sizes = np.array([np.array(pair) for pair in bbox_sizes])
    kmeans = KMeans(n_clusters=9, random_state=17).fit(bbox_sizes)
    anchors = kmeans.cluster_centers_

    xa = [int(anchor[0]) for anchor in anchors]
    ya = [int(anchor[1]) for anchor in anchors]

    plt.scatter(xa, ya, c="#FF0000")
    plt.title('Anchors')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()
    return anchors


def find_and_save_anchors(frames):
    '''
        Method for finding saving the best anchors for a list of frames.

        Args:
            frames:list<Frame> - the frames from which to get the width and height for calculating the best anchors
        Returns:
            anchors:list<int> - the best anchors for the given frames

        '''
    anchors = k_means_anchor_boxes(frames)
    anchors = sorted(anchors, key=lambda pair: iou(pair[0], pair[1], NETWORK_IMAGE_SIZE[0], NETWORK_IMAGE_SIZE[1]))

    print("Anchors:")
    for anchor in anchors:
        print(anchor, "sort for:", anchor[0] * anchor[1], "iou:",
              iou(anchor[0], anchor[1], NETWORK_IMAGE_SIZE[0], NETWORK_IMAGE_SIZE[1]))

    np.save('network_files/anchors.npy', anchors)
    return anchors


def create_ground_truths(frames, anchors):
    '''
        Method for creating the network input from labeled frames and anchors.

        Args:
            frames: list<Frame> - frames from which to get data
            anchors:list<int> - anchors to be used when calculating data for network
        Returns:
            networkY:list<ndarray> - the processed labels used by the network as ground truths

        '''
    print("Started creating ground truths.")

    # rows(Y) first, columns (X) last
    networkY = [np.zeros((len(frames), GRID_SIZE_SCALE_1[1], GRID_SIZE_SCALE_1[0], 24)),
                np.zeros((len(frames), GRID_SIZE_SCALE_2[1], GRID_SIZE_SCALE_2[0], 24)),
                np.zeros((len(frames), GRID_SIZE_SCALE_3[1], GRID_SIZE_SCALE_3[0], 24))]
    frameCounter = -1
    for frame in frames:
        if frameCounter % 100 == 0 and frameCounter != 0:
            print("Processed frames: " + str(frameCounter))
        frameCounter += 1
        for region in frame.regions:
            gridSize, anchorPosition, bestAnchor = get_anchor_for_region(region, anchors)
            # get the adequate cell of a region by grid size
            regionRow, regionColumn = region.getCellPositionByGridSize(gridSize)
            if frameCounter == 50:  # and regionRow == 0 and regionColumn == 50:
                a = ''
            # old values from the cell prediction where the region shall be placed math.log(gridSize // 13, 2)
            positionByGridSize = int(log(gridSize // 10, 2))
            cell_prediction = networkY[positionByGridSize][frameCounter][regionRow][regionColumn]
            # begining position of area to be written
            offset = anchorPosition * 8

            tx, ty, tw, th = calculate_network_bbox_parameters(region, bestAnchor, gridSize)
            # this one should be 1, although on the predicting end we run a sigmoid over it to reduce it to 0-1 interval
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

    print("Finished creating ground truths.")
    return networkY


def calculate_network_bbox_parameters(region, bestAnchor, gridSize):
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


def iou(regionWidth, regionHeight, anchorWidth, anchorHeight):
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


def get_anchor_for_region(region, anchors):
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
            iou(region.width * NETWORK_IMAGE_SIZE[0], region.height * NETWORK_IMAGE_SIZE[1], anchor[0], anchor[1]))
    gridSize = (2 ** (score.index(min(score)) // 3)) * 10
    anchorPosition = score.index(min(score)) % 3
    bestAnchor = anchors[score.index(min(score))]
    return gridSize, anchorPosition, bestAnchor


def display_statistics(frames):
    """
    Method for displaying various statistics for the given dataset

    Args:
        frames:Frame - dataset
    """
    noOfRegions = 0
    for frame in frames:
        noOfRegions += len(frame.regions)

    print("There are " + str(len(frames)) + " frames found")
    print("There are " + str(noOfRegions) + " regions found")

    noOfRedLights = 0
    noOfYellowLights = 0
    noOfGreenLights = 0
    for frame in frames:
        for region in frame.regions:
            if region.cls == "red":
                noOfRedLights += 1
            if region.cls == "yellow":
                noOfYellowLights += 1
            if region.cls == "green":
                noOfGreenLights += 1

    print("Number of red traffic lights: " + str(noOfRedLights))
    print("Number of yellow traffic lights: " + str(noOfYellowLights))
    print("Number of green traffic lights: " + str(noOfGreenLights))


def get_Frames_from_Driveu_images(images):
    """
    Method for mapping a list of DriveuImage to a list of Frame

    Args:
        images:list<DriveuImage> - Driveu images to be mapped
    Returns:
        frames:list<Frame> - Driveu images converted to frames

    """
    print("Started creating frames . . .")
    frames = []
    # Transform DriveuImages into frames
    for idx_d, img in enumerate(images):
        frames.append(map_DriveuImage_to_Frame(img))

        if len(frames) != 0 and len(frames) % 100 == 0:
            print("Frames processed:", len(frames))
    print("Finished creating frames.")
    return frames


def main(args):
    # Load database
    database = DriveuDatabase(args.label_file)
    print("Loading database . . .")
    database.open(args.image_file)
    print("Database loaded.")

    database.images = get_filtered_images(database.images, args.min_width, args.min_height)

    save_converted_images(database.images)

    frames = get_Frames_from_Driveu_images(database.images)

    name = args.label_file.split("/")[-1].split(".")[0]
    saveFilePath = args.output_file + "/frames_" + name + ".file"
    print("Started saving frames . . .")
    with open(saveFilePath, "wb") as file:
        pickle.dump(frames, file)
    print("Finished saving " + str(len(frames)) + " frames.")

    display_statistics(frames)

    # anchors are sorted decreasingly on iou with full image size
    anchors = find_and_save_anchors(frames)

    networkY = create_ground_truths(frames, anchors)

    print("Started saving ground truths.")
    with open("network_files/networkY.file", "wb", ) as file:
        pickle.dump(networkY, file, protocol=4)
    print("Finished saving ground truths")


if __name__ == "__main__":
    main(parse_args())
