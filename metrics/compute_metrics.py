import argparse
import os
import pickle

import cv2 as cv

from config import IMAGE_SIZE
from metrics.intersect_gt_and_dr import intersect_ground_truths_and_detection_results
from metrics.mAP import calculate_mAP
from sequence_display.cnn_yolo import YoloNetwork

import warnings
warnings.simplefilter("ignore", ResourceWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame_file", default="network_files/frames_fulda.file",
                        help='Location of frame file to be used')
    parser.add_argument("--image_file", default="images",
                        help='Location of the folder where all the image folders are contained')
    parser.add_argument("--output_file", default="../network_files",
                        help='Location where to store the metrics')
    return parser.parse_args()


def save_detection_results(imageFolder):
    """
        Method for saving the detection results after running predictions on the images in a given folder

        Args:
            imageFolder:string - folder from which to predict

        """
    print("Started saving detection results . . .")
    network = YoloNetwork('network_files/model.h5')
    count=0
    for filename in os.listdir(imageFolder):
        count+=1
        if count % 100 == 0:
            print("Detections saved:", count)

        img = cv.imread(os.path.join(imageFolder, filename))
        if img is not None:
            _, boxes = network.predict_one_image(img)
            save_one_detection(filename.split(".")[0], boxes)
    print("Saved detection results at metrics/detection_results.")



def save_one_detection(fileName, boxes):
    """
        Method for predicting on one image and saving the detection results

        Args:
            fileName:string - image on which to run the prediction
            boxes:list<BoundBox> - predicted boxes for the given image

        """
    # erase everything
    file = open("metrics/detection_results/" + fileName + ".txt", "w")
    file.close()

    with open("metrics/detection_results/" + fileName + ".txt", "a") as myfile:
        for box in boxes:
            line = box.actual_class + " "

            # for detection confidence
            line += str(box.score / 100) + " "

            # left
            line += str(box.xmin) + " "
            # top
            line += str(box.ymin) + " "
            # right
            line += str(box.xmax) + " "
            # bottom
            line += str(box.ymax) + " "
            line += "\n"

            myfile.write(line)


def save_gt_bboxes_in_map_format(frameFile):
    """
        Method for saving the coordinates of the ground truth boxes

        Args:
            frameFile:string - location of the frame file from which to load the ground truths

        """
    print("Saving ground truths . . .")

    frames = []
    with open(frameFile, "rb") as fp:  # Unpickling
        frames += pickle.load(fp)

    for frame in frames:
        # 'images/Fulda/DE_BBBR667_2015-05-15_17-51-09-037326_k0.jpg'
        fileName = frame.pathToImage.split("/")[-1].split(".")[0]
        # erase everything
        file = open("metrics/ground_truths/" + fileName + ".txt", "w")
        file.close()

        with open("metrics/ground_truths/" + fileName + ".txt", "a") as myfile:

            for region in frame.regions:
                line = region.cls + " "

                # left
                line += str(int(region.topLeftX * IMAGE_SIZE[0])) + " "
                # top
                line += str(int(region.topLeftY * IMAGE_SIZE[1])) + " "
                # right
                line += str(int(region.bottomRightX * IMAGE_SIZE[0])) + " "
                # bottom
                line += str(int(region.bottomRightY * IMAGE_SIZE[1])) + " "
                line += "\n"

                myfile.write(line)
    print("Ground truths saved in: metrics/ground_truths.")



if __name__ == "__main__":
    args = parse_args()

    save_detection_results(args.image_file)
    save_gt_bboxes_in_map_format(args.frame_file)

    intersect_ground_truths_and_detection_results()
    calculate_mAP()
