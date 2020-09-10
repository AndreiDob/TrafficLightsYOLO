import tensorflow as tf
from tensorflow.python.keras.backend import binary_crossentropy

LAMBDA_COORD = 1
LAMBDA_NO_OBJ = 1

iouTreshold = 0.7


def getIouMask(y_true, y_pred):
    """
        Method for selecting which predictions should have their loss ignored as they
        are close enough to the ground truth
        Details in bachelor paper: page 23

        Args:
            y_pred:list<ndarray> - network prediction
            y_true:list<ndarray> - ground truth
        Returns:
            mask:ndarray - the mask for which which ox losses should be kept/ignored

        """
    true_xy = tf.sigmoid(y_true[..., :2])
    true_wh = tf.math.exp(y_true[..., 2:4])

    pred_xy = tf.sigmoid(y_pred[..., :2])
    pred_wh = tf.math.exp(y_pred[..., 2:4])

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    # minimul de x si de y al fiecarei cutii
    # x and y minimum for each box
    pred_mins = pred_xy - pred_wh_half
    # maximul de x si de y al fiecarei cutii
    # x and y maximum for each box
    pred_maxes = pred_xy + pred_wh_half

    # selecteaza x si y fix ale punctului din stg sus al cutiei care reprezinta intersectia
    # select the x and y of the top left corner of the intersection box
    intersect_mins = tf.maximum(pred_mins, true_mins)
    # selecteaza x si y fix ale punctului din dreapta jos al cutiei care reprezinta intersectia
    # select the x and y of the bottom right corner of the intersection box
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    # calculeaza inaltimea si latimea cutiei de intersectie din cele 2 puncte obtinute anterior
    # computes the height and width of the intersection box using the previously computed points
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    # calculeaza suprafata zonei de intersectie
    # computes the surface of the intersection box
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    # calculeaza suprafata zonei de uniune
    # computes the surface of the union box
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    #selects which predictied boxes have an IOU with the ground truth bigger than the given threshold
    mask = tf.to_float(iou_scores < iouTreshold) + 0.1
    return mask


def custom_iou_loss(y_true, y_pred):
    """
        Method for calculating the loss from the predicted Y and the ground truth Y

        Args:
            y_pred:list<ndarray> - network prediction
            y_true:list<ndarray> - ground truth
        Returns:
            final_loss:float - the computed error for the current prediction

        """

    y_true = tf.verify_tensor_all_finite(y_true, "y_true not finite")
    y_pred = tf.verify_tensor_all_finite(y_pred, "y_pred not finite")

    shape = [tf.shape(y_pred)[k] for k in range(4)]
    y_true = tf.reshape(y_true, [shape[0], shape[1], shape[2], 3, 8])

    xy_true = y_true[..., :2]
    wh_true = y_true[..., 2:4]
    obj_true = y_true[..., 4]
    cls_true = y_true[..., 5:]

    y_pred = tf.reshape(y_pred, [shape[0], shape[1], shape[2], 3, 8])
    xy_pred = y_pred[..., :2]
    wh_pred = y_pred[..., 2:4]
    obj_pred = tf.sigmoid(y_pred[..., 4])
    cls_pred = tf.sigmoid(y_pred[..., 5:])

    iou_mask = getIouMask(y_true, y_pred)

    # square difference between all x in True and Pred for each anchor in each grid cell
    x_squared = tf.math.squared_difference(xy_true[..., 0], xy_pred[..., 0])

    # square difference between all y in True and Pred for each anchor in each grid cell
    y_squared = tf.math.squared_difference(xy_true[..., 1], xy_pred[..., 1])

    # ad x_sq and y_sq and multiply with 1 or 0 if there is an object there or not
    xy_sq_sum = tf.multiply(tf.math.add(x_squared, y_squared), obj_true)

    xy_sq_sum = tf.where(tf.is_nan(xy_sq_sum), tf.zeros_like(xy_sq_sum), xy_sq_sum)
    loss_xy = tf.reduce_sum(xy_sq_sum * iou_mask) * LAMBDA_COORD  # * LAMBDA_XY

    # square difference between all w in True and Pred for each anchor in each grid cell
    w_squared = tf.math.squared_difference(tf.sqrt(tf.abs(wh_true[..., 0])), tf.sqrt(tf.abs(wh_pred[..., 0])))
    # square difference between all h in True and Pred for each anchor in each grid cell
    h_squared = tf.math.squared_difference(tf.sqrt(tf.abs(wh_true[..., 1])), tf.sqrt(tf.abs(wh_pred[..., 1])))
    # ad w_sq and h_sq and multiply with 1 or 0 if there is an object there or not
    wh_sq_sum = tf.multiply(tf.math.add(w_squared, h_squared), obj_true)
    # the sum of x_sq and y_sq in all anchors and grid cells
    loss_wh = tf.multiply(tf.reduce_sum(wh_sq_sum * iou_mask), LAMBDA_COORD)

    confidence_loss_if_object = tf.reduce_sum(tf.multiply(binary_crossentropy(obj_true, obj_pred), obj_true))
    confidence_loss_if_not_object = tf.multiply(tf.reduce_sum(tf.multiply(binary_crossentropy(obj_true, obj_pred),
                                                                          tf.subtract(tf.ones_like(obj_true),
                                                                                      obj_true))), LAMBDA_NO_OBJ)

    class_loss = tf.reduce_sum(tf.multiply(tf.reduce_mean(binary_crossentropy(cls_true, cls_pred), axis=-1), obj_true))

    final_loss = loss_xy + loss_wh + confidence_loss_if_object + confidence_loss_if_not_object + class_loss

    return final_loss
