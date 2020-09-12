
System requirements

For training the network, the system requires a PC/laptop with a graphic
card. For testing it does not require much memory, but with more, time performance is improved.

In order to be able to install and run the training application and the
already-trained network, the system needs to have an Anaconda environment
that contains the following: Python 3.6, Tensorfow-gpu, Keras, Numpy, Pillow, Opencv and Matplotlib.

Installation

 Create Anaconda environment and install Python 3.6, Tensor
ow-gpu,
Keras, Numpy, Pillow, Opencv and Matplotlib.
 Install PyCharm and configure it to use the newly created environment

Run the system

Data preprocessing For preprocessing the data, a user needs to down-
load and unzip the Driveu dataset. Next, the user has to run the pre-
process labels.py script found in the preprocessing module.

This scrip requires the following command-line arguments:

 min width - Minimum width of the bounding boxes
 min height - Minimum height of the bounding boxes
 label - Location of label file to be used
 image - Location of the folder where all the image folders are
contained
 output - Location of the folder in which to store the processed
annotations

Training 

For training the system, a user has to open the train.py script
found in the training module, adjust the contained hyperparameters, then
run the script. The network will start training, periodically displaying its
progress.

The system takes care automatically of learning rate decrease, saving
the trained model and early stopping the training, in case of convergence.
After the training, the script displays the training/validation loss graph as
an evolution over time.

Predicting 

To start the prediction process, a user has to first plot the ROC
curve in order to select a good threshold for the confidence under which the
predicted bounding boxes are ignored. This can be achieved by running the
model visualization.py script found in the visualization module.

After selecting a threshold to be used as the confidence threshold, the
user has to open the predict.py script found in the predicting module. Here,
he has to set which part of the dataset he wants to run a prediction on, give
the previously selected threshold and run the script.

As the Tensorflow backend is not initialized yet, the system will take
about 7 seconds to initialize, after which the prediction will be done, the re-
sults post-processed and displayed. The user can move to the next predicted
image by pressing any key.

Running detection on a video

To run a detection on a video, the user
has to save each frame in a folder. Then the sequence predict.py script
found in the sequence display module has to be run.

The required command-line parameters are as follows:

 image folder - The folder from where to take the images for prediction
 delay - Delay between images in miliseconds. At least 1 ms should be
used.

Computing metrics 

To compute different metrics of the model, the user
has to run the compute metrics.py script found in the metrics module.

The required command-line parameters are as follows:

 frame - The location of frame file to be used for extracting the
ground truths
 image - The location of the folder where all the image folders are
contained. This is used for finding the detection results
 output - The location where to store the metrics

After running this script, the system will make predictions based on the
given images, save the detection results, save the ground truth labels, then
intersect the two of them. Then, the metrics are computed. The mean
average precision is printed and displayed, while the other metrics ca be
found in the given output.
