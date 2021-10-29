import os
import argparse
import time

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from numpy.lib.shape_base import dstack

import config
from utils import pred_helpers
from utils.evaluate_segmentation_functions import evaluate_segmentation
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

dataset = "val"
# get best epoch
save_results = "{}results/WGAN_{}/".format(config.toppath, opt.trial)
if opt.epoch == 9999:
    loss_g = np.load(save_results + "G_losses.npy")
    epoch = (
        np.argmin(loss_g[opt.nr_e_skipped :]) + opt.nr_e_skipped
    )  # neglect first 50 epochs because not converged yet
else:
    epoch = opt.epoch
print(epoch)
if opt.mult_trials:
    train_input = "gen_patches_DP_{}_{}_{}_{}".format(opt.trial, "all", epoch, opt.thresh)
else:
    train_input = "gen_patches_DP_{}_{}_{}".format(opt.trial, epoch, opt.thresh)

# loop over parameters
for augm in config.augmented:
    for bs in config.batch_size:
        for dp in config.dropout:
            for lr in config.learning_rates:
                for patient in config.PATIENTS[dataset]:
                    config_sess = tf.ConfigProto()
                    config_sess.gpu_options.allow_growth = True
                    config_sess.gpu_options.visible_device_list = config.vis_dev
                    set_session(tf.Session(config=config_sess))
                    pred_helpers.predict_and_save_prob_map(
                        patient,
                        dataset,
                        config.epochs_unet,
                        lr,
                        augm,
                        bs,
                        dp,
                        train_input,
                    )
                    K.clear_session()

print("DONE")


# EVALUATION
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
config_sess.gpu_options.visible_device_list = config.vis_dev
set_session(tf.Session(config=config_sess))

measures = "DICE,HDRFDST@0.95@"
patients_segm = config.PATIENTS[dataset]

# loop over each model
start_total = time.time()

# create results folder for evaluation segmentation
results_path = get_paths.get_results_path()
if not os.path.exists(results_path):
    os.makedirs(results_path)
executable_path = get_paths.get_executable_path()
print("executable_path: ", executable_path)
res_avg = np.empty(
    (
        len(config.augmented)
        * len(config.batch_size)
        * len(config.dropout)
        * len(config.learning_rates),
        2,
    )
)
res_sd = np.empty(
    (
        len(config.augmented)
        * len(config.batch_size)
        * len(config.dropout)
        * len(config.learning_rates),
        2,
    )
)
i = 0
best_idx = 0
for augm in config.augmented:
    for bs in config.batch_size:
        for dp in config.dropout:
            for lr in config.learning_rates:
                run_name = get_paths.get_run_name(
                    config.epochs_unet, bs, lr, dp, augm, train_input
                )
                csv_path = (
                    results_path
                    + "eval_segment_"
                    + dataset
                    + "_"
                    + time.strftime("%Y%m%d-%H%M%S")
                    + "_"
                    + run_name
                    + ".csv"
                )
                csv_path_per_patient = (
                    results_path
                    + "eval_segment_"
                    + dataset
                    + "_per_patient_"
                    + time.strftime("%Y%m%d-%H%M%S")
                    + ".csv"
                )
                mtrcs, sds = evaluate_segmentation(
                    patients_segm,
                    dataset,
                    config.epochs_unet,
                    bs,
                    lr,
                    dp,
                    augm,
                    train_input,
                    measures,
                    csv_path,
                    csv_path_per_patient,
                    executable_path,
                )

                # print(mtrcs.shape)
                res_avg[i] = np.array(mtrcs)
                res_sd[i] = np.array(sds)
                best_idx = np.argmax(res_avg[:, 0])
                print(res_avg[i], best_idx)

                if (best_idx == i) or (i == 0):
                    best_augm = augm
                    best_bs = bs
                    best_dp = dp
                    best_lr = lr
                i += 1

duration_total = int(time.time() - start_total)
print(
    "performance assessment took:",
    (duration_total // 3600) % 60,
    "hours",
    (duration_total // 60) % 60,
    "minutes",
    duration_total % 60,
    "seconds",
)
print("DONE")

# Best test set
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
config_sess.gpu_options.visible_device_list = config.vis_dev
set_session(tf.Session(config=config_sess))

# evaluate U-net (best val) on test set
dataset = "test"
patients_segm = config.PATIENTS[dataset]

for patient in config.PATIENTS[dataset]:
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.visible_device_list = config.vis_dev
    set_session(tf.Session(config=config_sess))
    pred_helpers.predict_and_save_prob_map(
        patient,
        dataset,
        config.epochs_unet,
        best_lr,
        best_augm,
        best_bs,
        best_dp,
        train_input,
    )
    K.clear_session()

# create results folder for evaluation segmentation
results_path = get_paths.get_results_path()
if not os.path.exists(results_path):
    os.makedirs(results_path)
executable_path = get_paths.get_executable_path()
print("executable_path: ", executable_path)
run_name = get_paths.get_run_name(
    config.epochs_unet, best_bs, best_lr, best_dp, best_augm, train_input
)
csv_path = (
    results_path
    + "eval_segment_"
    + dataset
    + "_"
    + time.strftime("%Y%m%d-%H%M%S")
    + "_"
    + run_name
    + ".csv"
)
csv_path_per_patient = (
    results_path
    + "eval_segment_"
    + dataset
    + "_per_patient_"
    + time.strftime("%Y%m%d-%H%M%S")
    + ".csv"
)
test_avg, test_sd = evaluate_segmentation(
    patients_segm,
    dataset,
    config.epochs_unet,
    best_bs,
    best_lr,
    best_dp,
    best_augm,
    train_input,
    measures,
    csv_path,
    csv_path_per_patient,
    executable_path,
)

eps_path = "{}results/WGAN_{}/epsilons.npy".format(
    config.toppath, opt.trial
)
if os.path.exists(eps_path):
    eps = np.load(eps_path)
    eps_current = eps[epoch]
else:
    eps_current = "unknown"

# write results in file
f = open("results/" + train_input + "_results.txt", "w")
f.write("Train input: " + train_input + "\n")
f.write("Best val DSC: {} ({})\n".format(res_avg[best_idx, 0], res_sd[best_idx, 0]))
f.write("Test DSC: {} ({})\n".format(test_avg[0], test_sd[0]))
f.write("Best U-net parameters: \n")
f.write(
    "Augm: {}, Batch size: {}, Dropout: {}, Learning rate: {}\n".format(
        best_augm, best_bs, best_dp, best_lr
    )
)
f.write("Epsilon: {}".format(eps_current))
f.close()
