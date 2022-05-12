# -*- coding = utf-8 -*-
# @Time : 2022/5/11 21:07
# @Author : 戎昱
# @File : sagan.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--g_num', type=int, default=5)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--g_conv_dim', type=int, default=64)
parser.add_argument('--d_conv_dim', type=int, default=64)
parser.add_argument('--lambda_gp', type=float, default=10)
parser.add_argument('--version', type=str, default='sagan_1')

# Training setting
parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
parser.add_argument('--d_iters', type=float, default=5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--g_lr', type=float, default=0.0001)
parser.add_argument('--d_lr', type=float, default=0.0004)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--b1', type=float, default=0.0)
parser.add_argument('--b2', type=float, default=0.9)

# using pretrained
parser.add_argument('--pretrained_model', type=int, default=None)

# Misc
parser.add_argument('--train', type=int, default=True)
parser.add_argument('--parallel', type=int, default=False)
parser.add_argument('--dataset', type=str, default='cifar', choices=['lsun', 'celeb'])
parser.add_argument('--use_tensorboard', type=int, default=False)

# Path
parser.add_argument('--image_path', type=str, default='./data')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--model_save_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./samples')
parser.add_argument('--attn_path', type=str, default='./attn')

# Step size
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--model_save_step', type=float, default=1.0)

parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

from sa_block import Self_Attn
from spectral import SpectralNorm

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 1, 4, 2, 1))
        last.append(nn.Conv2d(1, 1, (3,3), (2,2), 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 2))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(opt.batch_size,opt.imsize, opt.z_dim, opt.g_conv_dim)
discriminator = Discriminator(opt.batch_size,opt.imsize, opt.d_conv_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(torch.randn(real_imgs.size(0), opt.z_dim)).cuda()
    # Get labels ranging from 0 to n_classes for n rows
    gen_imgs,gf1,gf2 = generator(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        # print(real_imgs.size())
        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z = Variable(torch.randn(real_imgs.size(0), opt.z_dim)).cuda()
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs,gf1,gf2 = generator(z)


        # Loss measures generator's ability to fool the discriminator
        validity,df1,df2 = discriminator(gen_imgs)

        # hinge loss
        g_loss = - validity.mean()

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real,dr1,dr2 = discriminator(real_imgs)
        d_real_loss = torch.nn.ReLU()(1.0 - validity_real).mean()

        # Loss for fake images
        validity_fake,df1,df2 = discriminator(gen_imgs.detach())
        d_fake_loss = torch.nn.ReLU()(1.0 + validity_fake).mean()

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)