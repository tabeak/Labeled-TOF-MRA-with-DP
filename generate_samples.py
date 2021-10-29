import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from collections import OrderedDict
from opacus.utils.module_modification import convert_batchnorm_modules

import config as c
import model as md
import utils.utils as ut


parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=str, required=True, help="Trial number")
parser.add_argument(
    "--epoch", type=int, required=True, help="Epoch number, epoch 9999 for best epoch"
)
parser.add_argument(
    "--nr_e_skipped",
    type=int,
    default=50,
    help="Number of epochs that should be skipped, only for best epoch",
)
parser.add_argument(
    "--thresh", type=float, required=True, help="Threshold for binarizing the labels"
)
parser.add_argument(
    "--upsampling", type=bool, default=False, help="Usage of upsampling layer in G"
)
parser.add_argument(
    "--lrg", type=float, default=0.0001, help="learning rate for G, default=0.0001"
)
parser.add_argument("--ngf", type=int, default=96, help="number of filters G")
parser.add_argument("--kg", type=int, default=5, help="kernel size G")
parser.add_argument("--strg", type=int, default=2, help="strides G")
parser.add_argument("--padg", type=int, default=2, help="padding G")
parser.add_argument("--opg", type=int, default=1, help="output padding G")
parser.add_argument(
    "--seed", type=int, default=999, help="Set random seed, default: 999"
)
opt = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# for matplotlib
plt.switch_backend("agg")

# get best epoch
save_results = "{}results/WGAN_{}/".format(c.toppath, opt.trial)
if opt.epoch == 9999:
    loss_g = np.load(save_results + "G_losses.npy")
    epoch = (
        np.argmin(loss_g[opt.nr_e_skipped :]) + opt.nr_e_skipped
    )  # neglect first X epochs because not converged yet
else:
    epoch = opt.epoch

# to ensure it doesn't run partly on another gpu
torch.cuda.set_device(c.cuda_n[0])

# create name for generated images file
save_test_path = "{}results/test_images/gen_patches_DP_{}_{}_{}.npz".format(
    c.toppath, opt.trial, epoch, opt.thresh
)

# Device selection
device = torch.device(
    "cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available() and c.ngpu > 0) else "cpu"
)
print(device)

# ####Create generator object##### #
netG = md.Generator(ngf=opt.ngf, kg=opt.kg, strg=opt.strg, padg=opt.padg, opg=opt.opg, upsampling=opt.upsampling)
netG = convert_batchnorm_modules(netG).to(device)

# Setup same optimizers and parameters as the trial run being evaluated for G
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrg)

# Print the model
print(netG)
saved_model_path = "{}models/WGAN_{}/epoch_{}.pth".format(
    c.toppath, opt.trial, epoch
)
saved_params_dict = torch.load(saved_model_path)

netG.load_state_dict(saved_params_dict["Generator_state_dict"], strict=False)
optimizerG.load_state_dict(saved_params_dict["OptimizerG_state_dict"])

# number of noise images to
if c.noise_type == "uniform":
    test_noise = torch.empty(c.n_test_samples, c.nz, 1, 1).uniform_(-1, 1)
else:
    test_noise = torch.randn(c.n_test_samples, c.nz, 1, 1)

dataloader = data_utils.DataLoader(test_noise, batch_size=512, shuffle=False)

test_fake = torch.empty(c.n_test_samples, 2, c.image_size, c.image_size)
for i, data in enumerate(dataloader):
    noise = data.to(device)
    with torch.no_grad():
        if i != len(dataloader) - 1:
            test_fake[i * 512 : (i + 1) * 512] = netG(noise).detach().cpu()
        else:
            test_fake[i * 512 : c.n_test_samples] = netG(noise).detach().cpu()


# save generated images as jpeg

for i, fake in enumerate(test_fake):
    # hard thresholding for visualisation
    sample = fake.clone()
    sample[1][sample[1] > opt.thresh] = 1
    sample[1][sample[1] <= opt.thresh] = 0
    sample[0] = ut.rescale_unet(sample[0])  # rescaling back to 0-255
    test_fake[i] = sample

# save all generated images as npy compression
gan_img = test_fake[:, 0, :, :].cpu().numpy()
gan_label = test_fake[:, 1, :, :].cpu().numpy()
np.savez_compressed(
    save_test_path,
    img=gan_img[:, :, :, np.newaxis],
    label=gan_label[:, :, :, np.newaxis],
)
