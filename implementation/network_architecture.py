from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *


def _conv_block_simple(inp, filter, kernel, stride, id, bnorm=True, leaky=True):
    """
        Method for creating the fundamental building block of YOLO:
                Convolution + Batch_Normalization + LeakyReLU
        Args:
            inp:Layer - input layer of the block
            filter:int - number of filters for the convolution
            kernel:int - kernel size for the convolution
            stride:int - stride for the convolution
            id:int - block id
            bnorm:bool - whether to use Batch Normalization
            leaky:bool - whether to use Leaky ReLU
        Returns:
            x:Layer - the previous layers given as input plus the current block

        """
    x = inp
    if stride > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D(filter,
               kernel,
               strides=stride,
               padding='valid' if stride > 1 else 'same',  # peculiar padding as darknet prefer left and top
               name='conv_' + str(id),
               use_bias=False if bnorm else True
               )(x)
    if bnorm: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(id), momentum=0.99)(x)
    if leaky: x = LeakyReLU(alpha=0.1, name='leaky_' + str(id))(x)
    return x


def create_model_architecture():
    """
        Method for creating the network architecture.
        Details in the bachelor paper: page 22
        """
    input_image = Input(shape=(320, 832, 3))
    x = _conv_block_simple(input_image, filter=32, kernel=3, stride=2, id=1)

    x = _conv_block_simple(x, filter=64, kernel=3, stride=2, id=2)
    skip2 = x
    x = _conv_block_simple(x, filter=32, kernel=3, stride=1, id=3)
    x = _conv_block_simple(x, filter=64, kernel=3, stride=1, id=4)
    x = add([skip2, x])

    # ---------128-----------
    x = _conv_block_simple(x, filter=128, kernel=3, stride=2, id=5)
    skip5 = x
    x = _conv_block_simple(x, filter=64, kernel=1, stride=1, id=6)
    x = _conv_block_simple(x, filter=128, kernel=3, stride=1, id=7)
    x = add([skip5, x])
    skip15 = x
    x = _conv_block_simple(x, filter=64, kernel=1, stride=1, id=8)
    x = _conv_block_simple(x, filter=128, kernel=3, stride=1, id=9)
    x = add([skip15, x])

    # ---------256-----------
    x = _conv_block_simple(x, filter=256, kernel=3, stride=2, id=10)
    skip10 = x
    x = _conv_block_simple(x, filter=128, kernel=1, stride=1, id=11)
    x = _conv_block_simple(x, filter=256, kernel=3, stride=1, id=12)
    x = add([skip10, x])
    skip12 = x
    x = _conv_block_simple(x, filter=128, kernel=1, stride=1, id=13)
    x = _conv_block_simple(x, filter=256, kernel=3, stride=1, id=14)
    x = add([skip12, x])

    # ---------1024-----------
    x = _conv_block_simple(x, filter=1024, kernel=3, stride=2, id=15)
    skip15 = x
    x = _conv_block_simple(x, filter=512, kernel=1, stride=1, id=16)
    x = _conv_block_simple(x, filter=1024, kernel=3, stride=1, id=17)
    x = add([skip15, x])
    skip_for26 = x

    x = _conv_block_simple(x, filter=512, kernel=3, stride=1, id=18)
    x = _conv_block_simple(x, filter=256, kernel=1, stride=1, id=19)
    x = _conv_block_simple(x, filter=512, kernel=3, stride=1, id=20)
    x = Conv2D(filters=24, kernel_size=1, strides=1, padding='same', name='conv_21')(x)
    yolo1 = x

    x = skip_for26
    x = _conv_block_simple(x, filter=128, kernel=1, stride=1, id=22)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip10])

    skip_for52 = x

    x = _conv_block_simple(x, filter=256, kernel=3, stride=1, id=23)
    x = _conv_block_simple(x, filter=128, kernel=1, stride=1, id=24)
    x = _conv_block_simple(x, filter=256, kernel=3, stride=1, id=25)
    x = Conv2D(filters=24, kernel_size=1, strides=1, padding='same', name='conv_26')(x)
    yolo2 = x

    x = skip_for52
    x = _conv_block_simple(x, filter=128, kernel=1, stride=1, id=27)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip5])
    x = _conv_block_simple(x, filter=128, kernel=3, stride=1, id=28)
    x = _conv_block_simple(x, filter=64, kernel=1, stride=1, id=29)
    x = _conv_block_simple(x, filter=128, kernel=3, stride=1, id=30)
    x = Conv2D(filters=24, kernel_size=1, strides=1, padding='same', name='conv_31')(x)
    yolo3 = x

    model = Model(input_image, [yolo1, yolo2, yolo3])
    return model
