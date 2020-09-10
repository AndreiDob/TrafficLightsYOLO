import random
from math import floor, ceil

import cv2
import numpy as np

from data_holders.Frame import Frame
from data_holders.Region import Region
from config import NETWORK_IMAGE_SIZE


def get_list_of_translations(frame, translationCodes=[1]):  # 1-dr jos , 2-stg jos, 3-dr sus, 4-stg sus
    translated = []
    if 1 in translationCodes: translated.append(
        get_translated_frame(frame, 0.2, 0.1))  # stoplines sunt jos si se pierd daca pun 0.2
    if 2 in translationCodes: translated.append(get_translated_frame(frame, -0.2, 0.1))
    if 3 in translationCodes: translated.append(
        get_translated_frame(frame, 0.2, -0.1))  # semafoarele sunt sus si se pierd
    if 4 in translationCodes: translated.append(get_translated_frame(frame, -0.2, -0.1))

    return translated


def get_translated_frame(frame, x_offset, y_offset):  # positive or negative procents| Ex: -0.1  ,  0.5
    img = frame.loadedImage
    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32(
        [[1, 0, x_offset * NETWORK_IMAGE_SIZE[0]], [0, 1, y_offset * NETWORK_IMAGE_SIZE[1]]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    img_translation = np.expand_dims(img_translation, axis=-1)

    newRegions = []
    for region in frame.regions:
        newRegion = Region(region.cls,
                           region.centerX + x_offset,
                           region.centerY + y_offset,
                           region.width,
                           region.height
                           )
        # we do not add regions outside the frame
        if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
        newRegions.append(newRegion)

    return Frame(frame.pathToImage, newRegions, loadedImage=img_translation)


def get_list_of_crops(frame):
    crops = []
    for i in range(2):
        # sus stanga
        crops.append(get_crooped_frame(frame,
                                       leftX=random.uniform(0.0, 0.1),
                                       rightX=random.uniform(0.6, 1),
                                       topY=random.uniform(0.0, 0.1),
                                       bottomY=random.uniform(0.6, 1)
                                       ))
        # sus dreapta
        crops.append(get_crooped_frame(frame,
                                       leftX=random.uniform(0.0, 0.4),
                                       rightX=random.uniform(0.9, 1),
                                       topY=random.uniform(0.0, 0.1),
                                       bottomY=random.uniform(0.6, 1)
                                       ))
        # jos stanga
        crops.append(get_crooped_frame(frame,
                                       leftX=random.uniform(0.0, 0.1),
                                       rightX=random.uniform(0.6, 1),
                                       topY=random.uniform(0.0, 0.4),
                                       bottomY=random.uniform(0.9, 1)
                                       ))
        # jos dreapta
        crops.append(get_crooped_frame(frame,
                                       leftX=random.uniform(0.0, 0.4),
                                       rightX=random.uniform(0.9, 1),
                                       topY=random.uniform(0.0, 0.4),
                                       bottomY=random.uniform(0.9, 1)
                                       ))
    return crops


def get_list_of_zooms(frame):
    zooms = []
    for i in range(2):
        # center zoom in
        zoomFactor = random.uniform(0.05, 0.25)
        zooms.append(get_crooped_frame(frame,
                                       leftX=0.0 + zoomFactor,
                                       rightX=1 - zoomFactor,
                                       topY=0.0 + zoomFactor,
                                       bottomY=1 - zoomFactor,
                                       ))
        # center zoom out
        newSize = random.randint(250, 400)
        zooms.append(get_zoomed_out_frame(frame, newSize))
    return zooms


def get_crooped_frame(frame, leftX, rightX, topY, bottomY):  # percents of image size

    new_width = int((rightX - leftX) * NETWORK_IMAGE_SIZE[0])
    new_height = int((bottomY - topY) * NETWORK_IMAGE_SIZE[1])

    newRegions = []
    for region in frame.regions:
        newRegion = Region(region.cls,
                           (region.centerX - leftX) * NETWORK_IMAGE_SIZE[0] / new_width,
                           (region.centerY - topY) * NETWORK_IMAGE_SIZE[1] / new_height,
                           region.width * NETWORK_IMAGE_SIZE[0] / new_width,
                           region.height * NETWORK_IMAGE_SIZE[1] / new_height
                           )
        # we do not add regions outside the frame
        if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
        newRegions.append(newRegion)

    leftX = int(leftX * NETWORK_IMAGE_SIZE[0])
    rightX = int(rightX * NETWORK_IMAGE_SIZE[0])
    topY = int(topY * NETWORK_IMAGE_SIZE[1])
    bottomY = int(bottomY * NETWORK_IMAGE_SIZE[1])

    img = np.copy(frame.loadedImage)
    # numpy uses rows(y) first
    crop_img = img[topY:bottomY, leftX:rightX]
    crop_img = cv2.resize(crop_img, NETWORK_IMAGE_SIZE)
    #crop_img = np.expand_dims(crop_img, axis=-1)

    return Frame(frame.pathToImage, newRegions, loadedImage=crop_img)


def get_gaussian_noised_frame(frame, variation):
    img = frame.loadedImage
    row = img.shape[0]
    col = img.shape[1]
    mean = 0
    sigma = variation ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, 1))
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 1)

    # cv2.imshow('image', img)
    # cv2.imshow('image2', noisy)
    # k = cv2.waitKey(0)
    return Frame(frame.pathToImage, frame.regions, loadedImage=noisy)


def get_flipped_frames(frames):
    newFrames = []
    for frame in frames:
        flippedImg = cv2.flip(frame.loadedImage, 1)
        #flippedImg = np.expand_dims(flippedImg, axis=-1)

        newRegions = []
        for region in frame.regions:
            newRegion = Region(region.cls,
                               1 - region.centerX,
                               region.centerY,
                               region.width,
                               region.height
                               )
            # we do not add regions outside the frame
            if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
            newRegions.append(newRegion)
        newFrames.append(Frame(frame.pathToImage, newRegions, loadedImage=flippedImg))

    return newFrames


def get_exposure_augmented_frames(frames):
    newFrames = []
    for frame in frames:
        for i in range(1):
            exposureFactor = random.uniform(-0.3, 0.3)
            contrastFactor = random.uniform(0.6, 1.5)
            image = np.copy(frame.loadedImage)
            image[..., :] = contrastFactor * image[..., :] + exposureFactor
            image = np.clip(image, 0, 1)

            newRegions = []
            for region in frame.regions:
                newRegion = Region(region.cls,
                                   region.centerX,
                                   region.centerY,
                                   region.width,
                                   region.height
                                   )
                # we do not add regions outside the frame
                if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
                newRegions.append(newRegion)
            newFrames.append(Frame(frame.pathToImage, newRegions, loadedImage=image))

    return newFrames


def get_zoomed_out_frame(frame, newSizeX, newSizeY):  # ouput image size
    img = frame.loadedImage
    zoomed_img = cv2.resize(img, dsize=(newSizeX, newSizeY))
    paddingSmallX = floor((NETWORK_IMAGE_SIZE[0] - newSizeX) / 2)
    paddingLargeX = ceil((NETWORK_IMAGE_SIZE[0] - newSizeX) / 2)
    paddingSmallY = floor((NETWORK_IMAGE_SIZE[1] - newSizeY) / 2)
    paddingLargeY = ceil((NETWORK_IMAGE_SIZE[1] - newSizeY) / 2)
    zoomed_img = cv2.copyMakeBorder(zoomed_img, paddingSmallY, paddingLargeY, paddingSmallX, paddingLargeX,
                                    cv2.BORDER_CONSTANT)
    #zoomed_img = np.expand_dims(zoomed_img, axis=-1)
    factorX = newSizeX / NETWORK_IMAGE_SIZE[0]
    factorY = newSizeY / NETWORK_IMAGE_SIZE[1]
    newRegions = []
    for region in frame.regions:
        newRegion = Region(region.cls,
                           region.centerX * factorX + paddingSmallX / NETWORK_IMAGE_SIZE[0],
                           region.centerY * factorY + paddingSmallY / NETWORK_IMAGE_SIZE[1],
                           region.width * factorX,
                           region.height * factorY
                           )
        # we do not add regions outside the frame
        if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
        newRegions.append(newRegion)

    return Frame(frame.pathToImage, newRegions, loadedImage=zoomed_img)


def get_gaussian_noised_frames(frames):
    newFrames = []
    for frame in frames:

        row, col, _ = frame.loadedImage.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, 1))
        noisy = np.copy(frame.loadedImage) + gauss
        noisy = np.clip(noisy, 0, 1)

        newRegions = []
        for region in frame.regions:
            newRegion = Region(region.cls,
                               region.centerX,
                               region.centerY,
                               region.width,
                               region.height
                               )
            # we do not add regions outside the frame
            if newRegion.centerX < 0 or newRegion.centerY < 0 or newRegion.centerX > 1 or newRegion.centerY > 1: continue
            newRegions.append(newRegion)
        newFrames.append(Frame(frame.pathToImage, newRegions, loadedImage=noisy))

    return newFrames
