import pickle
import warnings

from config import IMAGE_SIZE
from utils.network_utils import get_split_shuffled_dataset, load_split_ground_truths

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn.metrics import roc_curve
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.optimizers import Adam

from implementation.custom_loss import custom_iou_loss
from scipy.special._ufuncs import expit
from training.train import LEARNING_RATE
from utils.cv_utils import load_resized_normalized_image
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utils import display_3d_data_with_hover


def visualize_filters():
    """
        Method for displaying the filters in a certain trained layer. Used for debugging

        """
    model = load_model('network_files/model_overfit.h5', compile=False)
    sgd = optimizers.SGD(lr=LEARNING_RATE)  # , clipvalue=0.5, clipnorm=1)
    model.compile(loss=custom_iou_loss, optimizer=sgd, metrics=["mse"])

    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        filters = layer.get_weights()[0]
        print(i, layer.name, filters.shape)

    filters = model.layers[228].get_weights()[0]
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    square = 6
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(filters[:, :, 0, ix - 1], cmap='gray')
            ix += 1
            if ix >= filters.shape[3]:
                break
                break
    # show the figure
    pyplot.show()


def visualize_feature_maps():
    """
        Method for displaying the feature maps from a certain trained layer. Used for debugging

        """

    model = load_model('network_files/model_bs8_valid.h5', compile=False)
    adam = Adam(lr=LEARNING_RATE, clipnorm=0.001)
    model.compile(loss=custom_iou_loss, optimizer=adam, metrics=["mse"])
    model.summary()

    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(i, layer.name, layer.output.shape)
    model = Model(inputs=model.inputs, outputs=model.layers[102].output)

    with open("network_files/imagePaths.txt", "rb") as fp:  # Unpickling
        filePaths = pickle.load(fp)

    image = load_resized_normalized_image(filePaths[0])
    image = np.expand_dims(image, axis=0)
    # show_one_image(image)

    feature_maps = model.predict(image)

    square = 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
            if ix >= feature_maps.shape[3] - 1:
                break
                break
    # show the figure
    pyplot.show()


def get_objectness_prediction(netout, for_true=False):
    """
        Method for getting the confidences of each predicted bounding box

        Args:
            netout:ndarray - network prediction
        Returns:
            obj_list:list<float> - confidences

        """
    obj_list = []
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    # from each cell prediction, it separates the 3 anchor box sections
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    sigmoid = expit
    objectness = netout[..., 4].flatten()
    if not for_true: objectness = sigmoid(objectness)
    # prediction = (objectness > objectnessThreshold).astype(int)
    for el in objectness:
        obj_list.append(el)
    return obj_list


def plot_roc_curve(no_of_examples=20, modelName="model", dataset="test"):
    """
        Method for plotting the ROC curve of a certain model on a certain dataset

        Args:
            no_of_examples:int - number of examples to run prediction on. This must me smaller than
                                the number of examples in the dataset
            modelName:string - name of the model from network_files to be used
            dataset:string - section of the dataset on which to run the prediction. Values= "train"/"valid"/"test"

        """
    model = load_model('network_files/' + modelName + '.h5', compile=False)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss=custom_iou_loss, optimizer=adam, metrics=[])

    Y_train, Y_valid, Y_test = load_split_ground_truths()
    trainFrames, validFrames, testFrames = get_split_shuffled_dataset()
    X_true = np.zeros((no_of_examples, IMAGE_SIZE[1], IMAGE_SIZE[0], 3))

    if dataset == "train":
        for i in range(no_of_examples):
            X_true[i] = load_resized_normalized_image(trainFrames[i].pathToImage)
        Y_true = [y[:no_of_examples] for y in Y_train]
        print("Started predicting X_train . . .")
    elif dataset == "valid":
        for i in range(no_of_examples):
            X_true[i] = load_resized_normalized_image(validFrames[i].pathToImage)
        Y_true = [y[:no_of_examples] for y in Y_valid]
        print("Started predicting X_valid . . .")
    elif dataset == "test":
        for i in range(no_of_examples):
            X_true[i] = load_resized_normalized_image(testFrames[i].pathToImage)
        Y_true = [y[:no_of_examples] for y in Y_test]
        print("Started predicting X_test . . .")
    else:
        raise (Exception("Wrong parameter toPredict. Choose train, valid or test"))

    Y_pred = model.predict(X_true)
    print("Finished predicting")

    obj_pred = list()
    for i in range(len(Y_pred)):
        # decode the output of the network
        obj_pred += get_objectness_prediction(Y_pred[i])

    obj_true = list()
    for i in range(len(Y_true)):
        # decode the output of the network
        obj_true += get_objectness_prediction(Y_true[i], for_true=True)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(obj_true, obj_pred)
    display_3d_data_with_hover(fpr_keras, tpr_keras, thresholds_keras,
                               x_label='False positive rate',
                               y_label='True positive rate'
                               )


def display_train_history(file="loss_history"):
    """
        Method for displaying the train and validation losses as a function of time

        Args:
            file:string - name of the file in which the loss history is stored

        """
    with open("network_files/" + file + ".file", "rb") as fp:
        history = pickle.load(fp)

    plt.plot(history[0])
    plt.plot(history[1])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # display_train_history(file="loss_history")
    plot_roc_curve(no_of_examples=3000, modelName="model", dataset="train")
