import _pickle as pickle
import os
import time

from keras.models import load_model
import nibabel as nib
import numpy as np

from utils import get_paths
from utils import dice_coef_loss, dice_coef


def load_nifti_mat_from_file(path_orig):
    """
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: String, path from where to load the nifti.
    :return: Nifti data as numpy array.
    """
    nifti_orig = nib.load(path_orig)
    return nifti_orig.get_data()  # transform the images into np.ndarrays


def create_and_save_nifti(mat, path_target):
    """
    Creates a nifti image from numpy array and saves it to given path.
    :param mat: Numpy array.
    :param path_target: String, path where to store the created nifti.
    """
    new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
    nib.save(new_nifti, path_target)  # save nifti to target dir
    print("New nifti saved to:", path_target)


def predict_and_save_prob_map(
    patient,
    dataset,
    epoch,
    lr,
    augm,
    batchsize,
    dropout,
    train_input,
    realonly=False,
):
    print("___________________________________________________________________")

    # check if file already exits
    run_name = get_paths.get_run_name(
        epoch, batchsize, lr, dropout, augm, train_input, realonly
    )
    new_filename = get_paths.get_prob_path(patient, run_name, dataset)
    print("Parameters of current run: ", run_name)

    if not os.path.isfile(new_filename):
        print("Patient number: ", patient)

        # -----------------------------------------------------------
        # LOADING MODEL, RESULTS AND WHOLE BRAIN MATRICES
        # -----------------------------------------------------------
        model_filepath = get_paths.get_model_path(run_name)

        model = load_model(
            model_filepath,
            custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef},
        )

        train_metadata_filepath = get_paths.get_train_metadata_path(run_name)

        try:
            with open(train_metadata_filepath, "rb") as handle:
                train_metadata = pickle.load(handle)

            patch_size = train_metadata["params"]["patch_size"]
            mean = train_metadata["params"]["mean"]
            sd = train_metadata["params"]["sd"]

            print("> Loading image...")
            # values between 0 and 255
            img_mat = load_nifti_mat_from_file(
                get_paths.get_original_data_path(dataset) + patient + "_img.nii.gz"
            )
            print("> Loading mask...")
            # values 0 and 1
            mask_mat = load_nifti_mat_from_file(
                get_paths.get_original_data_path(dataset) + patient + "_mask.nii.gz"
            )

            # -----------------------------------------------------------
            # PREDICTION
            # -----------------------------------------------------------
            # the segmentation is going to be saved in this probability matrix
            prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
            x_dim, y_dim, z_dim = prob_mat.shape

            # get the x, y and z coordinates where there is brain
            x, y, z = np.where(mask_mat)

            # get the z slices with brain
            z_slices = np.unique(z)

            # start cutting out and predicting the patches
            starttime_total = time.time()
            # proceed slice by slice
            for i in z_slices:
                slice_vox_inds = np.where(z == i)
                # find all x and y coordinates with brain in given slice
                x_in_slice = x[slice_vox_inds]
                y_in_slice = y[slice_vox_inds]
                # find min and max x and y coordinates
                slice_x_min = min(x_in_slice)
                slice_x_max = max(x_in_slice)
                slice_y_min = min(y_in_slice)
                slice_y_max = max(y_in_slice)

                # calculate number of predicted patches in x and y direction
                # in given slice
                num_of_x_patches = np.int(
                    np.ceil((slice_x_max - slice_x_min) / patch_size)
                )
                num_of_y_patches = np.int(
                    np.ceil((slice_y_max - slice_y_min) / patch_size)
                )

                # predict patch by patch in given slice
                for j in range(num_of_x_patches):
                    for k in range(num_of_y_patches):
                        # find the starting and ending x and y coordinates of
                        # given patch
                        patch_start_x = slice_x_min + patch_size * j
                        patch_end_x = slice_x_min + patch_size * (j + 1)
                        patch_start_y = slice_y_min + patch_size * k
                        patch_end_y = slice_y_min + patch_size * (k + 1)
                        # if the dimensions of the probability matrix are
                        # exceeded shift back the last patch
                        if patch_end_x > x_dim:
                            patch_end_x = slice_x_max
                            patch_start_x = slice_x_max - patch_size
                        if patch_end_y > y_dim:
                            patch_end_y = slice_y_max
                            patch_start_y = slice_y_max - patch_size

                        # get the patch with the found coordinates from the
                        # image matrix
                        img_patch = img_mat[
                            patch_start_x:patch_end_x, patch_start_y:patch_end_y, i
                        ]

                        # normalize the patch with mean and standard deviation
                        # calculated over training set
                        img_patch = img_patch.astype(np.float)
                        img_patch -= mean
                        img_patch /= sd

                        # predict the patch with the model and save to
                        # probability matrix
                        prob_mat[
                            patch_start_x:patch_end_x, patch_start_y:patch_end_y, i
                        ] = np.reshape(
                            model.predict(
                                np.reshape(img_patch, (1, patch_size, patch_size, 1)),
                                batch_size=1,
                                verbose=0,
                            ),
                            (patch_size, patch_size),
                        )

            # how long does the prediction take for a patient
            duration_total = time.time() - starttime_total
            print(
                "prediction in total took:",
                (duration_total // 3600) % 60,
                "hours",
                (duration_total // 60) % 60,
                "minutes",
                duration_total % 60,
                "seconds",
            )

            # -----------------------------------------------------------
            # SAVE AS NIFTI
            # -----------------------------------------------------------
            create_and_save_nifti(prob_mat, new_filename)
        except pickle.UnpicklingError:
            print("Not calculated")
