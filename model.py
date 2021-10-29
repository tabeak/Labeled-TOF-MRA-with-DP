import torch.nn as nn

import config as c


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("SpectralNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngf=96, kg=5, strg=2, padg=2, opg=1, upsampling=False):
        super(Generator, self).__init__()
        self.upsampling = upsampling
        self.ngf = ngf
        self.kg = kg
        self.strg = strg
        self.opg = opg
        self.padg = padg
        if kg==5:
            self.opg1 = 0
        elif kg==4:
            self.opg1 = 1
        if upsampling:
            self.linear1 = nn.Sequential(nn.Linear(c.nz, ngf * 16 * 3 * 3))
            self.convtrans1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 16, ngf * 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
            )
            # state size. (ngf*16) x 3 x 3
            self.convtrans2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 16, ngf * 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            )
            # state size. (ngf*8) x 6 x 6
            self.convtrans3 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 8, ngf * 4, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )
            # state size. (ngf*4) x 12 x 12
            self.convtrans4 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 4, ngf * 2, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            )
            # state size. (ngf) x 24 x 24
            self.convtrans5 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 2, ngf, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )
            # state size. (ngf) x 48 x 48
            self.convtrans6 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf, c.nc, kernel_size=5, stride=2, padding=2),
            )
            self.activationG = nn.Tanh()
        else:
            self.convtrans1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(c.nz, ngf * 16, kg, 2, 1, self.opg1, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
            )
            # state size. (ngf*16) x 3 x 3
            self.convtrans2 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, kg, strg, padg, opg, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            )
            # state size. (ngf*8) x 6 x 6
            self.convtrans3 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, kg, strg, padg, opg, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )
            # state size. (ngf*4) x 12 x 12
            self.convtrans4 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, kg, strg, padg, opg, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            )
            # state size. (ngf) x 24 x 24
            self.convtrans5 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, kg, strg, padg, opg, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )
            # state size. (ngf) x 48 x 48
            self.convtrans6 = nn.Sequential(
                nn.ConvTranspose2d(ngf, c.nc, kg, strg, padg, opg, bias=False)
            )
            self.activationG = nn.Tanh()
        # state size. (nc) x 96 x 96)


    def forward(self, inp):

        if self.upsampling:
            x = inp.view(inp.size()[0], -1)
            x = self.linear1(x)
            x = x.view(x.size()[0], self.ngf * 16, 3, 3)
            x = self.convtrans1(x)
        else:
            x = self.convtrans1(inp)
        x = self.convtrans2(x)
        x = self.convtrans3(x)
        x = self.convtrans4(x)
        x = self.convtrans5(x)
        last_conv_out = self.convtrans6(x)
        tanh_out = self.activationG(last_conv_out)
        return tanh_out


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ndf=96, kd=5, strd=2):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.kd = kd
        self.strd = strd

        self.conv1 = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(c.nc, ndf, kd, strd, strd, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf) x 48 x 48
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kd, strd, strd, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*2) x 24 x 24
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kd, strd, strd, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*4) x 12 x 12
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kd, strd, strd, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, kd, strd, strd, bias=False),
            nn.InstanceNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = nn.Conv2d(ndf * 16, 1, kd, strd, 1, bias=False)


    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        last_conv_output = self.conv5(x)
        sig_out = self.conv6(last_conv_output)
        return sig_out


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
