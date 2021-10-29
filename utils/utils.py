import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd

import time
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import keras.backend as K
from keras.models import Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Input
from keras.layers import UpSampling2D, concatenate, BatchNormalization

import config as c
from utils import get_paths


# #################################################### #
# ####### data pre-processing helper functions ####### #
# #################################################### #


def normalise(x):

    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def rescale_unet(x):

    return 255 * (x - x.min()) / (x.max() - x.min())


# train function for unet
def train_and_save(
    train_X,
    train_y,
    epoch,
    batchsize,
    lr,
    dropout,
    augm,
    train_input,
    mean_train,
    sd_train,
    realonly=False,
):

    # create the name of current run
    run_name = get_paths.get_run_name(
        epoch, batchsize, lr, dropout, augm, train_input, realonly
    )
    print("Parameters of current run: ", run_name)

    model_filepath = get_paths.get_model_path(run_name)
    train_metadata_filepath = get_paths.get_train_metadata_path(run_name)

    # if model does not exit, train it
    loading = False
    if not os.path.isfile(model_filepath):
        # -----------------------------------------------------------
        # CREATING MODEL
        # -----------------------------------------------------------
        model = get_unet(
            c.PATCH_SIZE,
            c.NUM_CHANNELS,
            c.ACTIVATION,
            c.FINAL_ACTIVATION,
            c.OPTIMIZER,
            lr,
            dropout,
            c.LOSS_FUNCTION,
            c.METRICS,
        )
        # model.summary()

        # -----------------------------------------------------------
        # CREATING DATA GENERATOR
        # -----------------------------------------------------------
        # transforming images and masks together
        data_gen_args = c.data_gen_args
        X_datagen = ImageDataGenerator(**data_gen_args)
        y_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to fit and flow methods
        seed = 1
        X_datagen.fit(train_X, augment=augm, seed=seed)
        y_datagen.fit(train_y, augment=augm, seed=seed)

        X_generator = X_datagen.flow(
            train_X, batch_size=batchsize, seed=seed, shuffle=True
        )
        y_generator = y_datagen.flow(
            train_y, batch_size=batchsize, seed=seed, shuffle=True
        )

        # combine generators into one which yields image and label
        train_generator = zip(X_generator, y_generator)

        # -----------------------------------------------------------
        # TRAINING MODEL
        # -----------------------------------------------------------
        start_train = time.time()
        # keras callback for saving the training history to csv file
        csv_logger = CSVLogger(get_paths.get_train_history_path(run_name))
        # training
        history = model.fit_generator(
            train_generator,
            validation_data=(train_X, train_y),
            steps_per_epoch=c.factor_train_samples * len(train_X) // batchsize,
            epochs=epoch,
            verbose=2,
            shuffle=True,
            callbacks=[csv_logger],
        )

        duration_train = int(time.time() - start_train)
        print(
            "training took:",
            (duration_train // 3600) % 60,
            "hours",
            (duration_train // 60) % 60,
            "minutes",
            duration_train % 60,
            "seconds",
        )

        # -----------------------------------------------------------
        # SAVING MODEL
        # -----------------------------------------------------------
        print("Saving model to ", model_filepath)
        model.save(model_filepath)

        print("Saving params to ", train_metadata_filepath)
        history.params["batchsize"] = batchsize
        history.params["dropout"] = dropout
        history.params["patch_size"] = c.PATCH_SIZE
        history.params["learning_rate"] = lr
        history.params["loss"] = c.LOSS_FUNCTION
        history.params["samples"] = c.factor_train_samples * len(train_X)
        history.params["total_time"] = duration_train
        history.params["augm"] = augm
        history.params["augm_params"] = data_gen_args
        if realonly:
            history.params["training_input"] = "real"
        else:
            history.params["training_input"] = train_input
        history.params["epoch"] = epoch
        history.params["mean"] = mean_train
        history.params["sd"] = sd_train
        results = {"params": history.params, "history": history.history}
        with open(train_metadata_filepath, "wb") as handle:
            pickle.dump(results, handle)

    else:
        print("Model already exists")


def conv_block(
    m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn
):
    """
    Bulding block with convolutional layers for one level.

    :param m: model
    :param num_kernels: number of convolution filters on the particular level,
    positive integer
    :param kernel_size: size of the convolution kernel, tuple of two positive
    integers
    :param strides: strides values, tuple of two positive integers
    :param padding: used padding by convolution, takes values: 'same' or 'valid'
    :param activation: activation_function after every convolution
    :param dropout: percentage of weights to be dropped, float between 0 and 1
    :param data_format: ordering of the dimensions in the inputs, takes values:
    'channel_first' or 'channel_last'
    :param bn: weather to use Batch Normalization layers after each convolution
    layer, True for use Batch Normalization,
     False do not use Batch Normalization
    :return: model
    """
    n = Convolution2D(
        num_kernels,
        kernel_size,
        strides=strides,
        activation=activation,
        padding=padding,
        data_format=data_format,
    )(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(rate=dropout)(n)
    n = Convolution2D(
        num_kernels,
        kernel_size,
        strides=strides,
        activation=activation,
        padding=padding,
        data_format=data_format,
    )(n)
    n = BatchNormalization()(n) if bn else n
    return n


def up_concat_block(m, concat_channels, pool_size, concat_axis, data_format):
    """

    :param m: model
    :param concat_channels: channels from left side onf Unet to be concatenated
    with the right part on one level
    :param pool_size: factors by which to downscale (vertical, horizontal),
    tuple of two positive integers
    :param concat_axis: concatenation axis, concatenate over channels, positive
    integer
    :param data_format: ordering of the dimensions in the inputs, takes values:
    'channel_first' or 'channel_last'
    :return: model"""
    n = UpSampling2D(size=pool_size, data_format=data_format)(m)
    n = concatenate([n, concat_channels], axis=concat_axis)
    return n


def get_unet(
    patch_size,
    num_channels,
    activation,
    final_activation,
    optimizer,
    learning_rate,
    dropout,
    loss_function,
    metrics=None,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    strides=(1, 1),
    num_kernels=None,
    concat_axis=3,
    data_format="channels_last",
    padding="same",
    bn=False,
):
    """
    Defines the architecture of the u-net. Reconstruction of the u-net
    introduced in: https://arxiv.org/abs/1505.04597

    :param patch_size: height of the patches, positive integer
    :param num_channels: number of channels of the input images, positive
    integer
    :param activation: activation_function after every convolution
    :param final_activation: activation_function of the final layer
    :param optimizer: optimization algorithm for updating the weights and bias
    values
    :param learning_rate: learning_rate of the optimizer, float
    :param dropout: percentage of weights to be dropped, float between 0 and 1
    :param loss_function: loss function also known as cost function
    :param metrics: metrics for evaluation of the model performance
    :param kernel_size: size of the convolution kernel, tuple of two positive
    integers
    :param pool_size: factors by which to downscale (vertical, horizontal),
    tuple of two positive integers
    :param strides: strides values, tuple of two positive integers
    :param num_kernels: array specifying the number of convolution filters in
    every level, list of positive integers
        containing value for each level of the model
    :param concat_axis: concatenation axis, concatenate over channels, positive
    integer
    :param data_format: ordering of the dimensions in the inputs, takes values:
    'channel_first' or 'channel_last'
    :param padding: used padding by convolution, takes values: 'same' or 'valid'
    :param bn: weather to use Batch Normalization layers after each convolution
    layer, True for use Batch Normalization,
     False do not use Batch Normalization
    :return: compiled u-net model
    """
    if metrics is None:
        metrics = ["accuracy"]
    if num_kernels is None:
        # num_kernels = [64, 128, 256, 512, 1024]
        num_kernels = [32, 64, 128, 256, 512]

    # specify the input shape
    inputs = Input((patch_size, patch_size, num_channels))

    # level 0
    conv_0_down = conv_block(
        inputs,
        num_kernels[0],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )
    pool_0 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_0_down)

    # level 1
    conv_1_down = conv_block(
        pool_0,
        num_kernels[1],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )
    pool_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_1_down)

    # level 2
    conv_2_down = conv_block(
        pool_1,
        num_kernels[2],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )
    pool_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_2_down)

    # level 3
    conv_3_down = conv_block(
        pool_2,
        num_kernels[3],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )
    pool_3 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_3_down)

    # level 4
    conv_4 = conv_block(
        pool_3,
        num_kernels[4],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )

    # UP-SAMPLING PART (right side of the U-net)

    # level 3
    concat_3 = up_concat_block(conv_4, conv_3_down, pool_size, concat_axis, data_format)
    conv_3_up = conv_block(
        concat_3,
        num_kernels[3],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )

    # level 2
    concat_2 = up_concat_block(
        conv_3_up, conv_2_down, pool_size, concat_axis, data_format
    )
    conv_2_up = conv_block(
        concat_2,
        num_kernels[2],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )

    # level 1
    concat_1 = up_concat_block(
        conv_2_up, conv_1_down, pool_size, concat_axis, data_format
    )
    conv_1_up = conv_block(
        concat_1,
        num_kernels[1],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )

    # level 0
    concat_0 = up_concat_block(
        conv_1_up, conv_0_down, pool_size, concat_axis, data_format
    )
    conv_0_up = conv_block(
        concat_0,
        num_kernels[0],
        kernel_size,
        strides,
        padding,
        activation,
        dropout,
        data_format,
        bn,
    )
    final_conv = Convolution2D(
        1,
        1,
        strides=strides,
        activation=final_activation,
        padding=padding,
        data_format=data_format,
    )(conv_0_up)

    # configure the learning process via the compile function
    model = Model(inputs=inputs, outputs=final_conv)
    model.compile(
        optimizer=optimizer(lr=learning_rate), loss=loss_function, metrics=metrics
    )
    print("U-net compiled.")

    # print out model summary to console
    model.summary()

    return model


def dice_coef(y_true, y_pred, smooth=0):
    """DICE coefficient

    Computes the DICE coefficient, also known as F1-score or F-measure.

    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :param smooth: Smoothing factor.
    :return: DICE coefficient of the positive class in binary classification.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """DICE loss function

    Computes the DICE loss function value.

    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :return: Negative value of DICE coefficient of the positive class in binary
    classification.
    """
    return -dice_coef(y_true, y_pred, 1)
