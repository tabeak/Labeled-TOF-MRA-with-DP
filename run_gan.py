import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchcsprng as prng
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import config as c
import model as md
import utils.utils as ut


# process input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--max_norm_dp", required=True, help="max gradient norm for DP")
parser.add_argument(
    "--clip_param_W", type=float, required=True, help="gradient clipping for WGAN"
)
parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
parser.add_argument("--ndf", type=int, default=96, help="number of filters D")
parser.add_argument("--ngf", type=int, default=96, help="number of filters G")
parser.add_argument("--kd", type=int, default=5, help="kernel size D")
parser.add_argument("--kg", type=int, default=5, help="kernel size G")
parser.add_argument("--strd", type=int, default=2, help="strides D")
parser.add_argument("--strg", type=int, default=2, help="strides G")
parser.add_argument("--padg", type=int, default=2, help="padding G")
parser.add_argument("--opg", type=int, default=1, help="output padding G")
parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs"
)
parser.add_argument(
    "--lrd", type=float, default=0.0001, help="learning rate for D, default=0.0001"
)
parser.add_argument(
    "--lrg", type=float, default=0.0001, help="learning rate for G, default=0.0001"
)
parser.add_argument("--trial", type=str, required=True, help="Trial number")
parser.add_argument(
    "--noisem", type=float, default=1.0, help="Noise multiplier (default 1.0)"
)
parser.add_argument(
    "--n_discr", type=int, default=1, help="Number of D updates per epoch"
)
parser.add_argument(
    "--upsampling", type=bool, default=False, help="Usage of upsampling layer in G"
)
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
torch.set_default_tensor_type("torch.cuda.FloatTensor")

# increase the speed of training if you are not varying the size of image after each epoch
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(c.cuda_n[0])
# for matplotlib plots
plt.switch_backend("agg")

# create necessary folders
save_model = "{}models/WGAN_{}/".format(c.toppath, opt.trial)
save_results = "{}results/WGAN_{}/".format(c.toppath, opt.trial)
if not os.path.isdir(save_model):
    os.mkdir(save_model)
if not os.path.isdir(save_results):
    os.mkdir(save_results)

# load numpy arrays with 41000 patches of size 96x96x1
data = np.load(c.toppath + "data/nr_patches_1000_random.npz")
imgs = data["img"]
mask = data["label"]

# normalise the input images to range between [-1,1]
imgs_norm = np.array([ut.normalise(i) for i in imgs[:, :, :, 0]])

# convert the images and masks to tensors
tensor_imgs = torch.FloatTensor(imgs_norm)
tensor_mask = torch.FloatTensor(
    mask[:, :, :, 0]
)  # removing channel dimension for mask or label as well
# stack them together for the generator as 2 channels
train_pair = torch.stack((tensor_imgs, tensor_mask), 1)
dataset = data_utils.TensorDataset(train_pair)

sample_gen = prng.create_random_device_generator("/dev/urandom")
dataloader = data_utils.DataLoader(
    dataset,
    generator=sample_gen,
    num_workers=c.workers,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(dataset),
        sample_rate=opt.batch_size / c.num_images,
        generator=sample_gen,
    ),
)

# Device selection
device = torch.device(
    "cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available() and c.ngpu > 0) else "cpu"
)


# build generator and initialize weights
netG = md.Generator(opt.ngf, opt.kg, opt.strg, opt.padg, opt.opg, opt.upsampling).to(device)
netG.apply(md.weights_init)

# create discriminator
netD = md.Discriminator(opt.ndf, opt.kd, opt.strd)
netD = convert_batchnorm_modules(netD)
netG = convert_batchnorm_modules(netG)
netD = netD.to(device)

# Print the model
print(netG)

# Handle multi-gpu if desired
if (device.type == "cuda") and (c.ngpu > 1):
    netD = nn.DataParallel(netD, c.cuda_n)
    netG = nn.DataParallel(netG, c.cuda_n)

# Apply the weights_init function to randomly initialize all weights
netD.apply(md.weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if c.noise_type == "uniform":
    fixed_noise = torch.empty(opt.batch_size, c.nz, 1, 1, device=device).uniform_(-1, 1)
else:
    fixed_noise = torch.randn(opt.batch_size, c.nz, 1, 1, device=device)


# Setup optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrd)
privacy_engine = PrivacyEngine(
    netD,
    sample_rate=opt.batch_size / c.num_images,
    alphas=c.alphas,
    noise_multiplier=opt.noisem,
    max_grad_norm=opt.max_norm_dp,
    secure_rng=c.secure_rng,
    target_delta=c.delta,
)
privacy_engine.attach(optimizerD)
torch.set_default_tensor_type("torch.FloatTensor")
epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(c.delta)
print(
    "(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
    % (epsilon, c.delta, best_alpha)
)
torch.set_default_tensor_type("torch.cuda.FloatTensor")

optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrg)

G_losses = []
D_losses = []
Wasserstein_d = []
real_p = []
fake_mp_b4update = []
fake_mp_afterupdate = []
epsilons = []

# Training Loop
# Lists to keep track of progress
img_list = []
iters = 0
duration = 0

nr_params_g = sum(p.numel() for p in netG.parameters())
nr_params_d = sum(p.numel() for p in netD.parameters())
print("nr params g:", nr_params_g)
print("nr params d:", nr_params_d)
print("Starting Training Loop...")

# check if model is already trained
path_last_epoch = save_model + "epoch_{}.pth".format(opt.epochs - 1)
if not os.path.exists(path_last_epoch):
    # For each epoch if model does not exist yet
    for epoch in range(opt.epochs):
        total_samp = 0
        epoch_start_time = time.time()
        # For each batch in the dataloader
        errD_iter = []
        errG_iter = []
        Wasserstein_d_iter = []
        D_x_iter = []
        D_G_z1_iter = []
        D_G_z2_iter = []

        for i, data in enumerate(dataloader, 0):
            errD_disc_iter = []
            Wasserstein_d_disc_iter = []
            D_x_disc_iter = []
            D_G_z1_disc_iter = []

            if (epoch == 0) and (i == 0):
                print(data[0].shape)

            for _ in range(opt.n_discr):

                # ########################## #
                # (1) Update D network
                # ######################### #
                # ## Train with all-real batch
                optimizerD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                # Forward pass real batch through D
                errD_real = netD(real_cpu)

                # Calculate loss on all-real batch
                errD_real = errD_real.view(-1).mean() * -1

                # Calculate gradients for D in backward pass
                errD_real.backward()
                optimizerD.step()

                # # Train with all-fake batch
                # Generate batch of latent vectors
                if c.noise_type == "uniform":
                    noise = torch.empty(b_size, c.nz, 1, 1, device=device).uniform_(
                        -1, 1
                    )  # uniform noise
                elif c.noise_type == "gaussian":
                    noise = torch.randn(b_size, c.nz, 1, 1, device=device)
                else:
                    noise = torch.empty(b_size, c.nz, 1, 1, device=device)
                    print(
                        "Please specify a valid distribution to sample noise vector from\n"
                    )

                # Generate fake image batch with G

                fake = netG(noise)
                if (epoch == 0) and (i == 0):
                    print(fake.shape)

                # Classify all fake batch with D
                errD_fake = netD(fake.detach())

                # Calculate D's loss on the all-fake batch
                errD_fake = errD_fake.view(-1).mean()

                # Calculate the gradients for this batch
                errD_fake.backward()
                optimizerD.step()

                if c.add_clip:
                    for parameter in netD.parameters():
                        parameter.data.clamp_(-opt.clip_param_W, opt.clip_param_W)

                # Update D
                errD = errD_fake + errD_real  # + gradient_penalty
                wasserstein = errD_fake + errD_real
                D_G_z1 = errD_fake.item()
                D_x = (errD_real * -1).item()

                # storing all the errors and outputs to be averaged over the discriminator iterations

                errD_disc_iter.append(errD.item())
                Wasserstein_d_disc_iter.append(wasserstein.item())
                D_G_z1_disc_iter.append(D_G_z1)
                D_x_disc_iter.append(D_x)

            # average all the errors and outputs of the whole n_disc iterations
            errD_disc_avg = np.mean(np.array(errD_disc_iter))
            Wasserstein_d_disc_avg = np.mean(np.array(Wasserstein_d_disc_iter))
            D_G_z1_disc_avg = np.mean(np.array(D_G_z1_disc_iter))
            D_x_disc_avg = np.mean(np.array(D_x_disc_iter))

            ############################
            # (2) Update G network
            ###########################

            # netG.zero_grad()
            optimizerG.zero_grad()

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_fake = netD(fake)

            errG = -output_fake.view(-1).mean()

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output_fake.view(-1).mean().item()
            # Update G
            optimizerG.step()

            errD_iter.append(errD_disc_avg)
            errG_iter.append(errG.item())

            D_x_iter.append(D_x_disc_avg)
            D_G_z1_iter.append(D_G_z1_disc_avg)
            D_G_z2_iter.append(D_G_z2)
            Wasserstein_d_iter.append(Wasserstein_d_disc_avg)

            iters += 1
            total_samp += b_size
            if i % 100 == 0:
                print("[%d/%d] samples done!\n" % (total_samp, len(dataset)))

        print(" End of Epoch %d \n" % epoch)

        # Output training stats averaged across batches after each epoch
        avg_errD = np.mean(np.array(errD_iter))
        avg_errG = np.mean(np.array(errG_iter))

        avg_D_x = np.mean(np.array(D_x_iter))
        avg_D_G_z1 = np.mean(np.array(D_G_z1_iter))
        avg_D_G_z2 = np.mean(np.array(D_G_z2_iter))

        avg_Wasserstein_d = np.mean(np.array(Wasserstein_d_iter))

        print(
            "[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
            % (epoch, opt.epochs, avg_errD, avg_errG, avg_D_x, avg_D_G_z1, avg_D_G_z2)
        )

        # Save Losses and outputs for plotting later
        G_losses.append(avg_errG.item())
        D_losses.append(avg_errD.item())

        real_p.append(avg_D_x)
        fake_mp_b4update.append(
            avg_D_G_z1
        )  # mean probability of classifying fake as real before updating D and G
        fake_mp_afterupdate.append(
            avg_D_G_z2
        )  # mean probability of classifying fake as real after updating D and G

        Wasserstein_d.append(avg_Wasserstein_d)

        np.save(save_results + "G_losses.npy", np.asarray(G_losses))
        np.save(save_results + "D_losses.npy", np.asarray(D_losses))

        np.save(save_results + "real_p.npy", np.asarray(real_p))
        np.save(save_results + "fake_mp_b4update.npy", np.asarray(fake_mp_b4update))
        np.save(
            save_results + "fake_mp_afterupdate.npy", np.asarray(fake_mp_afterupdate)
        )

        np.save(save_results + "Wasserstein_distance.npy", np.array(Wasserstein_d))

        # get privacy level
        torch.set_default_tensor_type("torch.FloatTensor")
        epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(c.delta)
        print(
            "(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
            % (epsilon, c.delta, best_alpha)
        )
        epsilons.append(epsilon)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Check how the generator is doing by saving G's output on fixed_noise
        # with torch.no_grad():
        fixed_fake = netG(fixed_noise).detach().cpu()

        img_list.append(fixed_fake.numpy())
        if opt.batch_size < 41:
            sample_idx = [0, 1, 2, 3]
        else:
            sample_idx = [10, 20, 30, 40]
        for idx in sample_idx:
            plt.figure()
            plt.imshow(
                torch.cat((fixed_fake[idx][0], fixed_fake[idx][1]), dim=1),
                cmap="gray",
                vmin=-1,
                vmax=1,
                animated=True,
            )
            plt.axis("off")
            plt.title("epsilon = %.2f" % (epsilon))
            plt.savefig(
                save_results
                + "fixed_fake_sample_%d_while_training_epoch_%d_.png" % (idx, epoch)
            )
            plt.close()

        # save model parameters'
        torch.save(
            {
                "Discriminator_state_dict": netD.state_dict(),
                "Generator_state_dict": netG.state_dict(),
                "OptimizerD_state_dict": optimizerD.state_dict(),
                "OptimizerG_state_dict": optimizerG.state_dict(),
            },
            save_model + "epoch_{}.pth".format(epoch),
        )

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_results + "losses.png")
        plt.close()

        np.save(save_results + "epsilons.npy", np.asarray(epsilons))
        plt.figure(figsize=(10, 5))
        plt.title("epsilon over epochs")
        plt.plot(epsilons)
        plt.xlabel("epoch")
        plt.ylabel("epsilon")
        plt.savefig(save_results + "eps.png")
        plt.close()

        epoch_end_time = time.time()

        duration = duration + (epoch_end_time - epoch_start_time)
        approx_time_to_finish = duration / (epoch + 1) * (opt.epochs - epoch)
        print(
            "Training time for epoch ",
            epoch,
            ": ",
            (epoch_end_time - epoch_start_time) / 60,
            " minutes.",
        )
        print(
            "Approximate time remaining for run to finish: ",
            approx_time_to_finish / 3600,
            " hours",
        )
