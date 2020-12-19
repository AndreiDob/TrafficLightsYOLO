# YOLO V3 inspired traffic light detection system

This project presents my own implementation of the YOLO system from the ground up. The idea was to put the basis of a real-time object-detection framework. In the beginning it closely followed the description in the [publication](https://pjreddie.com/media/files/papers/YOLOv3.pdf), but was later modified to better fit the problem of traffic light detection. 

The big YOLO V3 architecture is too resource intensive and lacks in prediction speed. To solve these problems, my contributions were: designing and implementing a new lighter, faster network architecture and improving the current loss function design. Also, I have created a special color selection system for when there is an uncertainty in traffic light classification.

**More details can be found in the [bachelor degree paper](BachelorThesis.pdf) that accompanies this project.**

# Results

By the use of the detection and classification capabilities of both YOLO
and the custom color selector, the system is capable of predicting traffic light
location and state with an incredibly good precision and in real time.
The Retinanet-101-500[1] network achieves 53.1 mAP at 11 frames per
second. The SSD321[2] network achieves 45.4 mAP at 16 frames per second.
My system is able to achieve 57.2 mAP, while running at 20 frames per
second, therefore outclassing some of the best state-of-the-art designs both
in terms of speed and accuracy.

# User manual

### System requirements
For training the network, the system requires a PC/laptop with a graphic card. For testing it does not require much memory, but with more, time performance is improved.

In order to be able to install and run the training application and the already-trained network, the system needs to have an Anaconda environment that contains the following: Python 3.6, Tensorfow-gpu, Keras, Numpy, Pillow, Opencv and Matplotlib.

### Installation
- Create Anaconda environment and install Python 3.6, Tensorflow-gpu, Keras, Numpy, Pillow, Opencv and Matplotlib.
- Install PyCharm and configure it to use the newly created environment

### Run the system
For preprocessing the data, a user needs to download and unzip the Driveu dataset. Next, the user has to run the preprocess labels.py script found in the preprocessing module.

This scrip requires the following command-line arguments:
- min width - Minimum width of the bounding boxes
- min height - Minimum height of the bounding boxes
- label - Location of label file to be used
- image - Location of the folder where all the image folders are contained
- output - Location of the folder in which to store the processed annotations

### Training 

For training the system, a user has to open the train.py script found in the training module, adjust the contained hyperparameters, then run the script. The network will start training, periodically displaying its progress.

The system takes care automatically of learning rate decrease, saving the trained model and early stopping the training, in case of convergence. After the training, the script displays the training/validation loss graph as an evolution over time.

### Predicting 
To start the prediction process, a user has to first plot the ROC curve in order to select a good threshold for the confidence under which the predicted bounding boxes are ignored. This can be achieved by running the model visualization.py script found in the visualization module.

After selecting a threshold to be used as the confidence threshold, the user has to open the predict.py script found in the predicting module. Here, he has to set which part of the dataset he wants to run a prediction on, give the previously selected threshold and run the script.

As the Tensorflow backend is not initialized yet, the system will take about 7 seconds to initialize, after which the prediction will be done, the re- sults post-processed and displayed. The user can move to the next predicted image by pressing any key.

### Running detection on a video
To run a detection on a video, the user has to save each frame in a folder. Then the sequence predict.py script found in the sequence display module has to be run.

The required command-line parameters are as follows:
- image folder - The folder from where to take the images for prediction
- delay - Delay between images in miliseconds. At least 1 ms should be used.

### Computing metrics 
To compute different metrics of the model, the user has to run the compute metrics.py script found in the metrics module.

The required command-line parameters are as follows:
- frame - The location of frame file to be used for extracting the ground truths
- image - The location of the folder where all the image folders are contained. This is used for finding the detection results
- output - The location where to store the metrics

After running this script, the system will make predictions based on the given images, save the detection results, save the ground truth labels, then intersect the two of them. Then, the metrics are computed. The mean average precision is printed and displayed, while the other metrics ca be found in the given output.
