import pickle
import random

from config import *
import numpy as np


def getCellDimensionsByGridSize(gridSize):
    if gridSize == 10:
        return CELL_WIDTH_10, CELL_HEIGHT_10
    elif gridSize == 20:
        return CELL_WIDTH_20, CELL_HEIGHT_20
    elif gridSize == 40:
        return CELL_WIDTH_40, CELL_HEIGHT_40
    else:
        raise Exception("Wrong grid size")


def load_frames():
    #in order do load with pickle, the directory of the classes Frame and Region have to be added to PythonPath
    frames = []
    with open("network_files/frames_Fulda_all.file", "rb") as fp:  # Unpickling
        frames += pickle.load(fp)
    return frames


def get_split_shuffled_dataset():
    frames = load_frames()
    random.Random(4).shuffle(frames)
    trainFrames = frames[0:int(TRAIN_RATIO * len(frames))]
    validFrames = frames[int(TRAIN_RATIO * len(frames)):int(VALID_RATIO * len(frames))]
    testFrames = frames[int(VALID_RATIO * len(frames)):]

    return trainFrames, validFrames, testFrames


def get_transformed_anchors(anchorsToTransformm=None):
    # [ [w,h] , [w,h] , [w,h] ...]
    if anchorsToTransformm != None:
        anchors = anchorsToTransformm
    else:
        anchors = np.load('network_files/anchors.npy')
    anchors13 = []
    anchors26 = []
    anchors52 = []
    for pair in anchors[0:3]:
        anchors13.append(pair[0])
        anchors13.append(pair[1])
    for pair in anchors[3:6]:
        anchors26.append(pair[0])
        anchors26.append(pair[1])
    for pair in anchors[6:9]:
        anchors52.append(pair[0])
        anchors52.append(pair[1])
    return [anchors13, anchors26, anchors52]


def split_train_test(input, trainRation, validRatio):
    datasetLength = input.shape[0]
    train = input[:int(datasetLength * trainRation)]
    valid = input[int(datasetLength * trainRation):int(datasetLength * validRatio)]
    test = input[int(datasetLength * validRatio):]
    return train, valid, test


def load_split_ground_truths():
    with open("network_files/networkY.file", "rb") as fp:  # Unpickling
        networkY = pickle.load(fp)

    Y_train1, Y_valid1, Y_test1 = split_train_test(networkY[0], TRAIN_RATIO, VALID_RATIO)
    Y_train2, Y_valid2, Y_test2 = split_train_test(networkY[1], TRAIN_RATIO, VALID_RATIO)
    Y_train3, Y_valid3, Y_test3 = split_train_test(networkY[2], TRAIN_RATIO, VALID_RATIO)

    return [Y_train1, Y_train2, Y_train3], [Y_valid1, Y_valid2, Y_valid3], [Y_test1, Y_test2, Y_test3]
