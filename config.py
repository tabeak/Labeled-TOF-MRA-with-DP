from keras.optimizers import Adam
from utils.utils import dice_coef_loss, dice_coef


""" 
GAN settings 
"""

# data related config
toppath = "/data/users/kossent/work/tabea/"  # path to data
nr_patients = 41
num_images = nr_patients * 1000
workers = 1 
image_size = 96

# training related config
nc = 2  # number of channels (image + segmentation label)
nz = 128  # size to noise vector (input G)
noise_type = "gaussian"  # gaussian or uniform
ngpu = 1  # nr of GPUs
cuda_n = [0]  # gpu number

# differentail privacy parameters
alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
delta = 1 / num_images
secure_rng = True
add_clip = True  # additional weight clipping

# GAN evaluation
n_test_samples = num_images  # how many images should be generated


""" 
U-net settings 
"""

# set paths
TOP_LEVEL = "/data/users/kossent/work/tabea/augmentation/error-based-GANs"
TOP_LEVEL_DATA = "/data/users/kossent/work/tabea/data/"

vis_dev = "0"  # GPU nr
nr_pats = 41
epochs_unet = 15
threshold_unet = 0.5  # for probabilities
batch_size = [64]
dropout = [0, 0.1]
learning_rates = [0.0001, 0.00008]
augmented = [True, False]

# for data augmentation
factor_train_samples = 2
data_gen_args = dict(
    rotation_range=30,
    horizontal_flip=False,
    vertical_flip=True,
    shear_range=20,
    width_shift_range=0,
    height_shift_range=0,
    fill_mode="constant",
)

# patient numbers
PATIENTS = {
    "train": [
        "0005", "0006", "0008", "0009",
        "0010", "0011", "0012", "0013",
        "0014", "0015", "0016", "0017",
        "0018", "0019", "0020", "0021",
        "0022", "0023", "0024", "0025",
        "0026", "0027", "0028", "0029",
        "0030", "0031", "0032", "0033",
        "0034", "0035", "0036", "0037",
        "0038", "0039", "0040", "0041",
        "0042", "0043", "0044", "0045",
        "0046",
    ],
    "val": [
        "0047", "0048", "0058", "0059",
        "0060", "0061", "0062", "0063",
        "0064", "0066", "0068",
    ],
    "test": [
        "0069", "0070", "0071", "0072",
        "0073", "0074", "0075", "0076",
        "0077", "0078", "0079", "0080",
        "0081", "0082",
    ],
}
NUM_CHANNELS = 1  # number of channels of the input images
ACTIVATION = "relu"
FINAL_ACTIVATION = "sigmoid"
LOSS_FUNCTION = dice_coef_loss
METRICS = [dice_coef, "accuracy"]
OPTIMIZER = Adam
NUM_PATCHES = 1000  # for one patient
UNET_TYPE = "halfunet"
PATCH_SIZE = 96
