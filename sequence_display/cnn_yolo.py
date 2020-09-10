import time

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.saving import load_model

from implementation.custom_loss import custom_iou_loss
from predicting.decoder_yolo_layer import decode_netout, correct_yolo_boxes, do_nms, get_boxes_with_good_classes, \
    combine_near_boxes
from config import IMAGE_SIZE, NETWORK_IMAGE_SIZE
import numpy as np
import cv2 as cv

from utils.cv_utils import show_one_image, draw_bbox, gamma_correct_image
from utils.network_utils import get_transformed_anchors

'''
This class is meant to be used in later developmnets of the project or 
for use in an independent project as a pre-trained predictor
'''


class YoloNetwork:
    """
            Class representing a trained network. This is meant to be used as is (with its dependencies, of course)
            in later projects.

            Attributes:
                anchors(list<int>):               anchors used in predicting
                model(Model):                     pretrained Keras model
                classTreshold(float):             confidence threshold
                nmsThreshold(float):              IOU threshold for non-max suppression
                gamma(float):                     gamma threshold for exposure correcting the input images
                delay(int):                       delay to put between predictions (in miliseconds)
                predictionTimes(list<float>):     list of time taken to predict each image. Used for statistics
            """
    def __init__(self, modelPath,delay):
        model = load_model(modelPath, compile=False)
        adam = Adam(lr=0.0001)
        model.compile(loss=custom_iou_loss, optimizer=adam, metrics=["mse"])
        # predicting a dummy image in order to initialize tensor flow backend. This takes about 7 seconds
        dummy_image = np.zeros((1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        model.predict(dummy_image)

        self.anchors = get_transformed_anchors()
        self.model = model
        self.classTreshold = 0.20
        self.nmsThreshold = 0.1
        self.gamma = 0.55
        self.delay=delay
        self.predictionTimes = []

    def predict_one_image(self, img):
        """
            Method for predicting on an image

            Args:
                img:ndarray - the image on which to predict
            Returns:
                img:ndarray - the processed image ready to be displayed
                boxes:list<BoundBox> - the boxes predicted for the given image

            """
        start_time = time.time()

        img = cv.resize(img, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA)
        img = gamma_correct_image(img, self.gamma)
        img = img / 255
        img = np.expand_dims(img, axis=0)

        yhat = self.model.predict(img)

        boxes = list()
        for scale in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[scale][0], self.anchors[scale], self.classTreshold, IMAGE_SIZE[1],
                                   IMAGE_SIZE[0])
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
        # suppress non-maximal boxes
        do_nms(boxes, self.nmsThreshold)
        combine_near_boxes(boxes, self.nmsThreshold)
        # define the labels
        labels = ["red", "yellow", "green"]
        # selects which classes are good
        boxes = get_boxes_with_good_classes(img[0], boxes, labels, self.classTreshold)

        self.predictionTimes.append(time.time() - start_time)
        if len(self.predictionTimes) % 100 == 0:
            print("Average prediction time for the last 100 frames: ",
                  (sum(self.predictionTimes) / len(self.predictionTimes)))

        return img[0], boxes

    def draw_predicted_image(self, image, boxes, zoomFactor):
        """
                Method for displaying a image and the predicted boxes on it

                Args:
                    image:ndarray - the image on which the prediction was run
                    boxes:list<BoundBox> - the boxes predicted by the network for this image
                    zoomFactor:int - by how much to zoom the image before displaying on a big screen

                """
        image = (image * 255)
        image = image.astype(np.uint8)
        image = cv.resize(image, dsize=(int(image.shape[1] * zoomFactor), int(image.shape[0] * zoomFactor)))
        for box in boxes:
            if abs(box.xmin) < 900 and abs(box.ymin) < 900 and abs(box.xmax) < 900 and abs(box.ymax) < 900:
                if box.actual_class == "red":
                    color = [0, 0, 255]
                elif box.actual_class == "yellow":
                    color = [0, 255, 255]
                elif box.actual_class == "green":
                    color = [0, 255, 0]
                else:
                    color = [255, 255, 255]
                draw_bbox(image,
                          int(box.xmin * zoomFactor),
                          int(box.ymin * zoomFactor),
                          int(box.xmax * zoomFactor),
                          int(box.ymax * zoomFactor),
                          str(int(box.score)),
                          color=color)
        show_one_image(image, sequenceDelay=self.delay)

    def _adjust_for_tiling(self, boxes):
        """
            Method for combining the predictions from 4 quarters of an image. This was used in the attempt to
            split the image and make 4 predictions on it in order to better detect small boxes

            Args:
                boxes:list<BoundBox> - all the boxes from the 4 quarters
            Returns:
                combined_boxes:list<BoundBox> - combined boxes from the 4 quarters

            """
        combined_boxes = []

        for box in boxes[0]:
            box.xmin /= 4
            box.ymin /= 4
            box.xmax /= 4
            box.ymax /= 4
            combined_boxes.append(box)

        for box in boxes[1]:
            box.xmin = 0.5 + box.xmin / 4
            box.ymin /= 4
            box.xmax = 0.5 + box.xmax / 4
            box.ymax /= 4
            combined_boxes.append(box)

        for box in boxes[2]:
            box.xmin = 0.5 + box.xmin / 4
            box.ymin = 0.5 + box.ymin / 4
            box.xmax = 0.5 + box.xmax / 4
            box.ymax = 0.5 + box.ymax / 4
            combined_boxes.append(box)

        for box in boxes[3]:
            box.xmin /= 4
            box.ymin = 0.5 + box.ymin / 4
            box.xmax /= 4
            box.ymax = 0.5 + box.ymax / 4
            combined_boxes.append(box)

        return combined_boxes

    def _get_tiled_image_sections(self, img):
        """
            Method for splitting the image into 4 quarters.This was used in the attempt to
            split the image and make 4 predictions on it in order to better detect small boxes

            Args:
                img:ndarray - image to be split
            Returns:
                tiled_image:ndarray - the split image represented as stacked quarters

            """

        height = img.shape[0]
        width = img.shape[1]

        img1 = img[0:int(height / 2), 0:int(width / 2)]
        img2 = img[0:int(height / 2), int(width / 2):width]
        img3 = img[int(height / 2):height, int(width / 2):width]
        img4 = img[int(height / 2):height, 0:int(width / 2)]

        tiled_image = np.zeros((4, NETWORK_IMAGE_SIZE[1], NETWORK_IMAGE_SIZE[0]))
        tiled_image[0] = cv.resize(img1, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA) / 255
        tiled_image[1] = cv.resize(img2, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA) / 255
        tiled_image[2] = cv.resize(img3, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA) / 255
        tiled_image[3] = cv.resize(img4, NETWORK_IMAGE_SIZE, interpolation=cv.INTER_AREA) / 255

        tiled_image = np.expand_dims(tiled_image, axis=-1)

        return tiled_image
