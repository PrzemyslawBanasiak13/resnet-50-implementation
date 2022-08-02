
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from keras.models import Model


def identity_block(x, f, filters):
    """
    x - input tensor
    f - kernel size for middle conv
    filters - list of numbers of filter for conv layers, eg. [32, 32, 32]

    returns:
    x - output of the identity block - tensor
    """

    x_short = x

    x = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    # add shortcut path to main path
    x = Add()([x, x_short])
    x = Activation('relu')(x)

    return x


def convolutional_block(x, f, filters, s=2):
    """
    x - input tensor
    f - kernel size for middle conv
    s - 1st main and short paths conv2d stride
    filters - list of numbers of filter for conv layers, eg. [32, 32, 32]

    returns:
    x - output of the conv block - tensor
    """

    x_short = x

    ### main path ###
    x = Conv2D(filters[0], (1, 1), strides=(s, s))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    ### short path ###
    x_short = Conv2D(filters[2], kernel_size=(1, 1), strides=(s, s), padding='valid')(x_short)
    x_short = BatchNormalization(axis=3)(x_short)

    # add shortcut path to main path
    x = Add()([x, x_short])
    x = Activation('relu')(x)

    return x


def get_model(input_shape=(256, 256, 3)):
    """
    input_shape - shape of the images of the dataset
    classes - integer, number of classes

    Returns:
    model - a Model() instance in Keras
    """

    i = Input(input_shape)
    x = ZeroPadding2D((3, 3))(i)

    # 1
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2
    x = convolutional_block(x, f=3, filters=[64, 64, 256], s=1)
    x = identity_block(x, 3, [256, 256, 256])
    x = identity_block(x, 3, [256, 256, 256])

    # 3
    x = convolutional_block(x, f=3, filters=[128, 128, 512], s=2)
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    # 4
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], s=2)
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    # 5
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], s=2)
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = AveragePooling2D((2, 2))(x)

    # output layer
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(101, activation='softmax')(x)

    # create model
    model = Model(inputs=i, outputs=x, name='ResNet-50')

    return model
