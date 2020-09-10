import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from implementation.custom_loss import custom_iou_loss
from training.data_generators import BatchGenerator
from implementation.network_architecture import create_model_architecture
from utils.network_utils import get_split_shuffled_dataset


LEARNING_RATE = 0.001


def train_save_model():
    """
        Method for training and saving the trained model

        """
    model = create_model_architecture()
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss=custom_iou_loss, optimizer=adam, metrics=[])

    # callbacks
    rlrop = ReduceLROnPlateau(monitor='val_loss', min_delta=0.1, factor=0.5, patience=3, verbose=1, mode='min', )
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=6, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(
        filepath="network_files/model.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_freq='epoch'
    )

    # get frames used for training and validation
    trainFrames, validFrames, _ = get_split_shuffled_dataset()
    print("Training on", len(trainFrames), "frames.")
    # load anchors used for generating the ground truths
    anchors = np.load('network_files/anchors.npy')

    BATCH_SIZE = 4
    training_generator = BatchGenerator(trainFrames, anchors, BATCH_SIZE, showImages=True, useAugmentation=False)
    validation_generator = BatchGenerator(validFrames, anchors, BATCH_SIZE, showImages=False, useAugmentation=False)

    history = model.fit(training_generator, validation_data=validation_generator, epochs=40, batch_size=BATCH_SIZE,
                        verbose=1,
                        callbacks=[rlrop, early_stop, checkpoint]
                        )

    model.save('network_files/model.h5')
    train_loss_history = history.history["loss"]
    valid_loss_history = history.history["val_loss"]
    losses = [train_loss_history, valid_loss_history]

    with open("network_files/loss_history.file", "wb") as fp:
        pickle.dump(losses, fp)


if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        train_save_model()
        # display_train_history()
