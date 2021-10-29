import argparse

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

import config
from utils.utils import train_and_save
from utils import get_paths


config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
config_sess.gpu_options.visible_device_list = config.vis_dev
set_session(tf.Session(config=config_sess))

# fix random seed for reproducibility
np.random.seed(7)
# set numpy to print only 3 decimal digits for neatness
np.set_printoptions(precision=9, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=str, required=True, help="Trial number")
parser.add_argument("--mult_trials", type=bool, default=False)
parser.add_argument(
    "--epoch", type=int, required=True, help="Epoch number, epoch 9999 for best epoch"
)
parser.add_argument(
    "--thresh", type=float, required=True, help="Threshold for binarizing the labels"
)
parser.add_argument(
    "--nr_e_skipped",
    type=int,
    default=50,
    help="Number of epochs that should be skipped, only for best epoch",
)
opt = parser.parse_args()

# get best epoch
save_results = "{}results/WGAN_{}/".format(config.toppath, opt.trial)
if opt.epoch == 9999:
    loss_g = np.load(save_results + "G_losses.npy")
    epoch = (
        np.argmin(loss_g[opt.nr_e_skipped :]) + opt.nr_e_skipped
    )  # neglect first 50 epochs because not converged yet
else:
    epoch = opt.epoch

if opt.mult_trials:
    for i in [2,3,4,5,6]:
        train_input = "gen_patches_DP_{}_{}_{}_{}".format(opt.trial, i, epoch, opt.thresh)
        gen_data_path = (
            config.toppath + "results/test_images/" + train_input + ".npz"
        )
        if i==2:
            train_tmp = np.load(gen_data_path)
            train_img_tmp = train_tmp["img"][:(41000//5)]
            train_label_tmp = train_tmp["label"][:(41000//5)]
        else:
            train_tmp = np.load(gen_data_path)
            train_img_tmp = np.concatenate((train_img_tmp, train_tmp["img"][:(41000//5)]), axis=0)
            train_label_tmp = np.concatenate((train_label_tmp, train_tmp["label"][:(41000//5)]), axis=0)
    mean_train = np.mean(train_img_tmp)
    sd_train = np.std(train_img_tmp)
    train_X = (train_img_tmp - mean_train) / sd_train
    train_y = train_label_tmp
    train_input = "gen_patches_DP_{}_{}_{}_{}".format(opt.trial, "all", epoch, opt.thresh)
else:
    train_input = "gen_patches_DP_{}_{}_{}".format(opt.trial, epoch, opt.thresh)
    gen_data_path = (
        config.toppath + "results/test_images/" + train_input + ".npz"
    )
    train_raw = np.load(gen_data_path)
    mean_train = np.mean(train_raw["img"])
    sd_train = np.std(train_raw["img"])
    train_X = (train_raw["img"] - mean_train) / sd_train
    train_y = train_raw["label"]

print("Training input size: ", train_X.shape, train_y.shape)

# loop over parameters
for augm in config.augmented:
    for bs in config.batch_size:
        for dp in config.dropout:
            for lr in config.learning_rates:
                config_sess = tf.ConfigProto()
                config_sess.gpu_options.allow_growth = True
                config_sess.gpu_options.visible_device_list = config.vis_dev
                set_session(tf.Session(config=config_sess))
                train_and_save(
                    train_X,
                    train_y,
                    config.epochs_unet,
                    bs,
                    lr,
                    dp,
                    augm,
                    train_input,
                    mean_train,
                    sd_train,
                )
                K.clear_session()

print("DONE")
