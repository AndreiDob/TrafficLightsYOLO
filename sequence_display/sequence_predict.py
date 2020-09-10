import argparse

import cv2 as cv
import os

from sequence_display.cnn_yolo import YoloNetwork


def predict_images_from_folder(folder, delay):
    """
        Method for predicting on the images from a given folder.

        Args:
            folder:string - path to the folder from which to load the images
            delay:int - how many miliseconds to pause between frame prediction.
                        This is used to give the user enough time to see that the model predicted correctly.
                        It is most useful when the video saved as frames in the folder is at a lower frame rate
                        than the network can predict. If run with delay=1ms, then the video would be too fast.

        """
    network = YoloNetwork('network_files/model.h5', delay)
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            image, boxes = network.predict_one_image(img)
            network.draw_predicted_image(image, boxes, zoomFactor=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default="images/Fulda",
                        help='The folder from where to take the images for prediction.')
    parser.add_argument("--delay", type=int, default="300",
                        help='Delay between images in miliseconds. At least 1 ms should be used.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_images_from_folder(args.image_folder, args.delay)
