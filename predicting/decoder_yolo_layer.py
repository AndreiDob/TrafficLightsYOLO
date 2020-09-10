import warnings

from data_holders.BoundingBox import BoundBox
from utils.math_utils import clamp

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn.cluster import KMeans
import time
import numpy as np
from scipy.special._ufuncs import expit
import cv2 as cv
from config import IMAGE_SIZE
from utils.cv_utils import draw_bbox, show_one_image


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    """
        Method for decoding(post-processing) the network output and transforming it in potential bounding boxes.

        Args:
            netout:ndarray - result of running a prediction on a set of images(Y)
            anchors:list<int> - list of anchors to be used for post-processing
            obj_thresh:float - confidence threshold
            net_h:int - network input height
            net_w:int - network input width
        Returns:
            boxes:list<BoundBox> - list of potential bounding boxes containing dimensions as percentages of the image size

        """

    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    # from each cell prediction, it separates the 3 anchor box sections
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    boxes = []

    # tx si ty  | expit = sigmoid
    sigmoid = expit
    netout[..., :2] = sigmoid(netout[..., :2])
    # skips tw and th
    # applies sigmoid on objectivity and classes
    netout[..., 4:] = sigmoid(netout[..., 4:])
    # on 4 is the objectivity
    # here each class prediction is multiplied by the objectivity of the cell-anchor
    # 13 13 3 [1]
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    # daca obj<treshold face toate clasele 0
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    start_time = time.time()
    # vectorization
    # here we prepare the things to be added/multiplied with x,y,w,h
    column_vect = np.array([range(0, grid_w)] * grid_h)
    column_vect = np.expand_dims(column_vect, axis=-1)
    column_vect = np.repeat(column_vect, 3, axis=-1)

    row_construct = np.transpose(np.array([range(0, grid_h)]))
    row_vect = np.tile(row_construct, grid_w)
    row_vect = np.expand_dims(row_vect, axis=-1)
    row_vect = np.repeat(row_vect, 3, axis=-1)

    anchor_width_construct = np.array([anchors[0], anchors[2], anchors[4]])
    anchor_width_construct = np.expand_dims(anchor_width_construct, axis=0)
    anchor_width_vect = np.tile(anchor_width_construct, [grid_h, grid_w, 1])

    anchor_height_construct = np.array([anchors[1], anchors[3], anchors[5]])
    anchor_height_construct = np.expand_dims(anchor_height_construct, axis=0)
    anchor_height_vect = np.tile(anchor_height_construct, [grid_h, grid_w, 1])

    # here we decode x,y,w,h
    x_vect = netout[..., 0]
    x_vect = np.divide(np.add(x_vect, column_vect), grid_w)

    y_vect = netout[..., 1]
    y_vect = np.divide(np.add(y_vect, row_vect), grid_h)

    w_vect = netout[..., 2]
    w_vect = np.divide(np.multiply(np.exp(w_vect), anchor_width_vect), net_w)

    h_vect = netout[..., 3]
    h_vect = np.divide(np.multiply(np.exp(h_vect), anchor_height_vect), net_h)

    classes_vect = netout[..., 5:]

    obj_vect = netout[..., 4]

    # flatten everything to run the obj mask
    x_flat = x_vect.flatten()
    y_flat = y_vect.flatten()
    w_flat = w_vect.flatten()
    h_flat = h_vect.flatten()
    classes_flat = classes_vect.reshape(-1, classes_vect.shape[-1])
    obj_flat = obj_vect.flatten()

    # create and apply the obj mask
    obj_mask = obj_flat > obj_thresh
    if obj_mask.max() == False:
        return []

    # apply mask and transform to pixels
    x_good = x_flat[obj_flat > obj_thresh]
    y_good = y_flat[obj_flat > obj_thresh]
    w_good = w_flat[obj_flat > obj_thresh]
    h_good = h_flat[obj_flat > obj_thresh]
    classes_good = classes_flat[obj_flat > obj_thresh]
    obj_good = obj_flat[obj_flat > obj_thresh]

    for i in range(x_good.shape[0]):
        x = x_good[i]
        y = y_good[i]
        w = w_good[i]
        h = h_good[i]
        cls = classes_good[i]
        obj = obj_good[i]
        box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, obj, cls)
        boxes.append(box)

    return boxes


def correct_yolo_boxes(boxes, image_w, image_h):
    """
        Method for translating the box coordinates from percents to actual image pixels

        Args:
            boxes:list<BoundinBox> - boxes to be transformed
            image_w:int - actual image width in pixels
            image_h:int - actual image height in pixels

        """
    for i in range(len(boxes)):
        boxes[i].xmin = clamp(int(boxes[i].xmin * image_w), 0, IMAGE_SIZE[0])
        boxes[i].xmax = clamp(int(boxes[i].xmax * image_w), 0, IMAGE_SIZE[0])
        boxes[i].ymin = clamp(int(boxes[i].ymin * image_h), 0, IMAGE_SIZE[1])
        boxes[i].ymax = clamp(int(boxes[i].ymax * image_h), 0, IMAGE_SIZE[1])


def _interval_overlap(interval_a, interval_b):
    """
        Method for calculating the intersection of two intervals

        Args:
            interval_a:list<int> - first interval
            interval_b:list<int> - second interval
        Returns:
            intersection of the two intervals

        """
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    """
        Method for calculating the IOU of two boxes

        Args:
            box1:BoundBox - first box
            box2:BoundBox - second box
        Returns:
            IOU of the two boxes

        """
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    if union == 0:
        return 1
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    """
        Method for supressing non-maximal bounding boxes

        Args:
            boxes:list<BoundBox> - bounding boxes to be grouped and suppressed
            nms_thresh:float - IOU threshold to be used for non-max suppression
        Returns:
            frames:list<Frame> - Driveu images converted to frames

        """
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def combine_near_boxes(boxes, iou_thershold):
    """
        Method for combining close boxes with different classes into a box whith
        all the classes contained in the starting boxes

        Args:
            boxes:list<BoundBox> - bounding boxes to be grouped
            iou_thershold:float - IOU threshold to be used for selecting grouping boxes

        """
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if bbox_iou(boxes[i], boxes[j]) >= iou_thershold:
                for cls in range(3):  # nb_classes
                    if boxes[i].classes[cls] != 0:
                        boxes[j].classes[cls] = boxes[i].classes[cls]
                        boxes[i].classes[cls] = 0


def get_boxes_with_good_classes(image, boxes, labels, thresh):
    """
        Method for checking which class is suitable for each bbox

        Args:
            image:ndarray- the image the prediction was run on. This is used in case that
                            the system decides to employ the color selector
            boxes:list<BoundBox> - bounding boxes to be considered
            labels:list<string> - actual names of the labels
            thresh:float - class threshold to be used for selecting which classes are relevant
        Returns:
            goodBoxes:list<BoundBox> - list of final detection boxes

        """
    cvThreshold = 0.2
    goodBoxes = []
    # enumerate all boxes
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] < thresh:
                box.classes[i] = 0
        sorted_indices = np.argsort([-box.classes[i] for i in range(len(box.classes))])
        if box.classes[sorted_indices[0]] == 0:
            continue
        if box.classes[sorted_indices[0]] - box.classes[sorted_indices[1]] > cvThreshold or \
                box.classes[sorted_indices[1]] == 0:
            box.actual_class = labels[sorted_indices[0]]
            box.score = box.classes[sorted_indices[0]] * 100
        else:
            cls_index = get_box_class_cv(image, box, sorted_indices[0], sorted_indices[1])
            box.actual_class = labels[cls_index]
            box.score = box.classes[cls_index] * 100
        goodBoxes.append(box)
    return goodBoxes


def get_bgr_from_index(index):
    """
        Method for translating the index of a label to its BGR color

        Args:
            index:int - label index( 0-red, 1-yellow, 2-green)
        Returns:
            ndarray containing the BGR color encoding

        """
    if index == 0:
        return np.array([0, 0, 255])
    elif index == 1:
        return np.array([0, 255, 255])
    elif index == 2:
        return np.array([0, 255, 0])


def get_box_class_cv(image, box, colorIdx1, colorIdx2):
    """
        Method for selecting which color to use for a box from two network-predicted colors
        Details in bachelor paper: page 23-24

        Args:
            image:ndarray - the image that contains the box
            box:BoundBox - the bounding box for which to decide the color
            colorIdx1:int - color label index( 0-red, 1-yellow, 2-green)
            colorIdx2:int - color label index( 0-red, 1-yellow, 2-green)
        Returns:
            frames:list<Frame> - Driveu images converted to frames

        """
    # First we crop the image to the dimensions indicated by the bounding box.
    # More exactly, we take the image from the predicted bounding box
    padding = 0
    image = (image * 255)
    image = image.astype(np.uint8)
    x_start = clamp(box.xmin - padding, 0, IMAGE_SIZE[0])
    x_stop = clamp(box.xmax + padding, 0, IMAGE_SIZE[0])
    y_start = clamp(box.ymin - padding, 0, IMAGE_SIZE[1])
    y_stop = clamp(box.ymax + padding, 0, IMAGE_SIZE[1])
    img = image[y_start:y_stop, x_start:x_stop]

    # flatten the bounding box image array to make it suitable for K-means
    colorArray = img.reshape((-1, 3))
    colorArray = np.float32(colorArray)

    # set the initial centroids to black and white
    initialCenters = np.array([[0, 0, 0], [255, 255, 255]])

    # run K-means on the bounding box image
    kmeans = KMeans(n_clusters=2, init=initialCenters, n_init=1).fit(colorArray)
    center = kmeans.cluster_centers_
    center = np.uint8(center)

    # Select the centroid of the lighter image. This centroid coresponds to the detected color in the bounding box.
    # The other centroid refers to the black outline of the traffic light
    sum = np.sum(center, axis=-1)
    maxPos = np.argmax(sum)
    color = center[maxPos]

    # get the network color predictions. We need to choose between the two of them.
    choice1 = get_bgr_from_index(colorIdx1)
    choice2 = get_bgr_from_index(colorIdx2)

    # Calculate the error between each choice and the detected color. Basically, this
    # tells us "how far" from the detected color each choice is
    err1 = np.sum(np.abs(np.subtract(color, choice1)))
    err2 = np.sum(np.abs(np.subtract(color, choice2)))

    # Select the choice with the smallest error (the color closest to the detected color)
    if err1 > err2:
        return colorIdx2
    else:
        return colorIdx1


def draw_boxes_from_loaded_image(image, boxes):
    """
        Method for displaying a image and the predicted boxes on it

        Args:
            image:ndarray - the image on which the prediction was run
            boxes:list<BoundBox> - the boxes predicted by the network for this image

        """
    image = (image * 255)
    image = image.astype(np.uint8)
    for box in boxes:
        if abs(box.xmin) < 900 and abs(box.ymin) < 900 and abs(box.xmax) < 900 and abs(box.ymax) < 900:
            # save_box_contents(image, box)
            # BlueGreenRed
            if box.actual_class == "red":
                color = [0, 0, 255]
            elif box.actual_class == "yellow":
                color = [0, 255, 255]
            elif box.actual_class == "green":
                color = [0, 255, 0]
            else:
                color = [255, 255, 255]

            draw_bbox(image,
                      int(box.xmin),
                      int(box.ymin),
                      int(box.xmax),
                      int(box.ymax),
                      str(int(box.score)),
                      color=color)
    show_one_image(image)
