from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.saving import load_model
import numpy as np

from implementation.custom_loss import custom_iou_loss
from predicting.decoder_yolo_layer import decode_netout, correct_yolo_boxes, do_nms, get_boxes_with_good_classes, \
    draw_boxes_from_loaded_image, combine_near_boxes
from config import NETWORK_IMAGE_SIZE, IMAGE_SIZE
from training.train import LEARNING_RATE
from utils.cv_utils import load_resized_normalized_image
from utils.network_utils import get_transformed_anchors, load_split_ground_truths, get_split_shuffled_dataset

from visualization.model_visualization import plot_roc_curve


def predict(no_of_examples, modelName="model", classTreshold=0.5, nmsThreshold=0.5, toPredict="test",
            predictGrTruth=False):
    """
        Method for predicting a set of images. By selecting predictGrTruth=True, the user can run the
        post-processing phase on top of the preprocessed data. As they are complementary, if the bounding boxes
        and images are displayed as they should, that means that both the encoder and the decoder work as required

        Args:
            no_of_examples:int - number of examples to run prediction on. This must me smaller than
                                the number of examples in the dataset
            modelName:string - name of the model from network_files to be used
            classTreshold:float - confidence threshold to use. Any detection with conf lower than this will be ignored
            nmsThreshold:float - maximum IOU of two images befor non-max suppression is applied
            toPredict:string - section of the dataset on which to run the prediction. Values= "train"/"valid"/"test"
            predictGrTruth:bool - whether to run the post-processing directly on the preprocessed data rather than
                                    on a network prediction. This is used to test the pre and post processing
        """

    Y_train, Y_valid, Y_test = load_split_ground_truths()
    trainFrames, validFrames, testFrames = get_split_shuffled_dataset()

    input_w, input_h = NETWORK_IMAGE_SIZE
    networkX = np.zeros((no_of_examples, IMAGE_SIZE[1], IMAGE_SIZE[0], 3))

    if toPredict == "train":
        for i in range(no_of_examples):
            networkX[i] = load_resized_normalized_image(trainFrames[i].pathToImage)
        print("Started predicting X_train . . .")
    elif toPredict == "valid":
        for i in range(no_of_examples):
            networkX[i] = load_resized_normalized_image(validFrames[i].pathToImage)
        print("Started predicting X_valid . . .")
    elif toPredict == "test":
        for i in range(no_of_examples):
            networkX[i] = load_resized_normalized_image(testFrames[i].pathToImage)
        print("Started predicting X_test . . .")
    else:
        raise (Exception("Wrong parameter toPredict. Choose train, valid or test"))

    model = load_model('network_files/' + modelName + '.h5', compile=False)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss=custom_iou_loss, optimizer=adam, metrics=["mse"])

    if predictGrTruth:
        if toPredict == "train":
            yhat = Y_train
        elif toPredict == "valid":
            yhat = Y_valid
        else:
            yhat = Y_test
    else:
        yhat = model.predict(networkX)

    print("Finished predicting")

    anchors = get_transformed_anchors()

    for frameNumber in range(yhat[0].shape[0]):
        boxes = list()
        for scale in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[scale][frameNumber], anchors[scale], classTreshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
        # suppress non-maximal boxes
        do_nms(boxes, nmsThreshold)
        combine_near_boxes(boxes, nmsThreshold)
        # define the labels
        labels = ["red", "yellow", "green"]
        # selects which classes are good
        boxes = get_boxes_with_good_classes(networkX[frameNumber], boxes, labels, classTreshold)
        # draw what we found
        draw_boxes_from_loaded_image(networkX[frameNumber], boxes)


if __name__ == "__main__":
    '''
    If you do not use the same anchors as the ones the model was trained on, 
    although the objects will be correctly localized, the width and height will be bad. 
    The model in network_files was trained on anchors_good_model.npy form the same model.
    Should you want to test it, just rename that file to anchors.npy and rerun the predict. 
    Or better, test it on some traffic light images by using sequence_display.
    
    ###### It can also be used for testing the pre and post processing. See documentation ######
    '''
    predict(4, classTreshold=0.01, modelName="model", nmsThreshold=0.1, toPredict="train", predictGrTruth=False)
    plot_roc_curve(no_of_examples=35, modelName="model", dataset="test")
