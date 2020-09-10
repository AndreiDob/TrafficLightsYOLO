from os import path

import cv2 as cv
import numpy as np
import sys

sys.path.append('..')

from config import NETWORK_IMAGE_SIZE, gammaCorrectionValue

def show_one_image(img, sequenceDelay=0):
    cv.imshow('image', img)
    k = cv.waitKey(sequenceDelay)

def draw_bbox(img, x1, y1, x2, y2, cls, color=[255, 0, 0]):
    # draw rectangle
    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

    # draw the class text
    text_s = cv.getTextSize(cls, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv.rectangle(img, (x2, y2), (x2 + text_s[0], y2 - text_s[1]), color, -1)
    cv.putText(img, cls, (x2, y2), cv.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1, )


def draw_point(img, x, y, color=(0, 0, 255)):
    cv.circle(img, (int(x), int(y)), 1, color, 1)


def draw_grid(img, noOfRows, noOfColumns, color=[0, 0, 0]):
    height = img.shape[0]
    width = img.shape[1]

    for x in range(0, width, width // noOfColumns):
        cv.line(img, (x, 0), (x, height), color, thickness=1)
    for y in range(0, height, height // noOfRows):
        cv.line(img, (0, y), (width, y), color, thickness=1)




def show_bboxes_and_image_centers(imagePathOrData, imageSize, regionsSimplified, showFromLoadedImage=False):
    if showFromLoadedImage:
        img = imagePathOrData
    else:
        img = cv.imread(imagePathOrData, cv.IMREAD_UNCHANGED)
    # resize image to 832x320
    img = cv.resize(img, imageSize, interpolation=cv.INTER_AREA)
    draw_grid(img, 10, 26)

    # draw the bboxes for each region
    for region in regionsSimplified:
        draw_bbox(img,
                  int(region.topLeftX * imageSize[0]),
                  int(region.topLeftY * imageSize[1]),
                  int(region.bottomRightX * imageSize[0]),
                  int(region.bottomRightY * imageSize[1]),
                  str(region.grid10Row) + " " + str(region.grid10Column))
        draw_point(img, region.centerX * imageSize[0], region.centerY * imageSize[1])
    show_one_image(img)



def load_resized_normalized_image(imagePath):
    img = cv.imread(imagePath, cv.IMREAD_UNCHANGED)
    if not img is not None:
        raise Exception("Error: Image couldn't be read propperly")
    height, width = img.shape[0], img.shape[1]
    if width == 0 or height == 0:
        raise Exception("Error: Image width or height is 0")
    img = cv.resize(img, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA)
    img = gamma_correct_image(img, gammaCorrectionValue)
    img = img.astype(np.float32) / 255
    return img



def gamma_correct_image(img, gamma):
    img = (img / 255) ** gamma * 255
    img = img.astype(np.uint8)
    return img
