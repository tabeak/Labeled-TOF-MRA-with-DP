import config


def get_real_patches_path():
    return config.TOP_LEVEL + "/Unet/train_data/nr_patches_1000_random.npz"


def get_sampled_real_patches_path(pathname):
    return config.TOP_LEVEL + "/Unet/train_data/" + pathname + ".npz"


# get all important parameters for current run
def get_run_name(
    epoch, batchsize, lr, dropout, augm, train_input, gen_and_real=False, realonly=False
):
    if realonly:
        return (
            config.UNET_TYPE
            + "_epochs_"
            + str(epoch)
            + "_batchsize_"
            + str(batchsize)
            + "_lr_"
            + str(lr)
            + "_dropout_"
            + str(dropout)
            + "_augm_"
            + str(augm)
            + "_real_"
            + str(config.nr_pats)
        )
    else:
        return (
            config.UNET_TYPE
            + "_epochs_"
            + str(epoch)
            + "_batchsize_"
            + str(batchsize)
            + "_lr_"
            + str(lr)
            + "_dropout_"
            + str(dropout)
            + "_augm_"
            + str(augm)
            + "_"
            + train_input
            + "_genandreal_"
            + str(gen_and_real)
        )


# path for unet model
def get_model_path(run_name):
    return config.TOP_LEVEL + "/Unet/models/" + run_name + ".h5"


# path for metadata of model
def get_train_metadata_path(run_name):
    return config.TOP_LEVEL + "/Unet/models/" + "metadata_" + run_name + ".pkl"


def get_train_history_path(run_name):
    return config.TOP_LEVEL + "/Unet/models/" + "train_history_" + run_name + ".csv"


# path to the probabilites maps
def get_prob_path(patient, run_name, dataset):
    return (
        config.TOP_LEVEL
        + "/Unet/results/probs_"
        + dataset
        + "/"
        + patient
        + "_"
        + run_name
        + ".nii"
    )


def get_original_data_path(dataset):
    return config.TOP_LEVEL_DATA + "PEGASUS/" + dataset + "/"


def get_result_xml_path(patient, run_name, dataset):
    return (
        config.TOP_LEVEL
        + "/Unet/results/eval_segment/"
        + dataset
        + "/"
        + patient
        + "_"
        + run_name
        + ".xml"
    )


def get_result_table_path(run_name, dataset):
    return (
        config.TOP_LEVEL
        + "/Unet/results/eval_segment/"
        + dataset
        + "/"
        + run_name
        + ".csv"
    )


def get_results_path():
    return config.TOP_LEVEL + "/Unet/results/eval_segment/"


def get_executable_path():
    return config.TOP_LEVEL + "/Unet/utils/EvaluateSegmentation"
