import argparse
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import torch

with open('./time_point', 'rb') as f:
    time_point = pickle.load(f)
# for p in time_point:
#     os.makedirs('F:/inversion_para/final_ions_gan/u6+/' + p + 'd', exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=156, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
#
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.bn = nn.Sequential(nn.BatchNorm2d(1),)
        self.label1 = nn.Sequential(nn.Conv2d(2, 64, 4, 2, 1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 32
        self.label2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2,inplace=True))  # 16
        self.label3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 8
        self.label4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 4
        self.label5 = nn.Sequential(nn.Conv2d(512, 1024, 4, 1, 0), nn.LeakyReLU(0.2,inplace=True))  # 1

        self.ct1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 1, 0), nn.BatchNorm2d(512, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 4
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 8
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 16
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh(), )  # 64

    def forward(self, well, label):
        label = torch.concat([well, label], dim=1)
        label1 = self.label1(label)
        label2 = self.label2(label1)
        label3 = self.label3(label2)
        label4 = self.label4(label3)
        label5 = self.label5(label4)

        ct1 = self.ct1(label5)
        input_2 = torch.concat([label4, ct1], dim=1)
        ct2 = self.ct2(input_2)
        input_3 = torch.concat([label3, ct2], dim=1)
        ct3 = self.ct3(input_3)
        input_4 = torch.concat([label2, ct3], dim=1)
        ct4 = self.ct4(input_4)
        input_5 = torch.concat([label1, ct4], dim=1)
        ct5 = self.ct5(input_5)
        return ct5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(1), )  # 只进行批归一化
        self.label1 = nn.Sequential(
                                    nn.Conv2d(32, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 16
                                    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 8
                                    nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 4
                                    nn.Conv2d(512, 1024, 4, 1, 0), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25),)  # 1
        self.pd = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.img_label = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),)  # 32

    def forward(self, well, img, label):
        img = torch.concat([well, img, label], dim=1)
        img = self.img_label(img)
        out = self.label1(img)
        out = out.view(out.shape[0], -1)
        validity = self.pd(out)
        return validity
# Loss function
adversarial_loss = torch.nn.BCELoss()
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
# Configure data loader
k_input = []
x_train = []
for m in range(200):
    k_batch = []
    x_batch = []
    for t_p in time_point:
        k_data = np.array(pd.read_csv('D:/surrogate_model/program/dcgan/cDCGAN/cdcgan-again/64x64_final/data/k_data/第' + str(m) + '个场.txt', header=None, sep=' '))
        '''The data size is about a few GB, only the data at 650 d is uploaded here, 
                for full ions training data please contact the author.'''
        with open('./data/ion_data/'+t_p+'d/no_'+str(m)+'_'+t_p+'天化学结果.pkl', 'rb') as f:
            ion_data = pickle.load(f)
            ion_data = np.where(ion_data < 0.000000001, 0.0, ion_data)  # lower bounds
        k_batch.append(k_data)
        x_batch.append(ion_data)
    k_batch_min = np.min(k_batch)
    k_batch_max = np.max(k_batch)
    k_batch_normal = (k_batch-k_batch_min)/(k_batch_max-k_batch_min)
    x_batch_min = np.min(x_batch)
    x_batch_max = np.max(x_batch)
    x_batch_normal = (x_batch-x_batch_min)/(x_batch_max-x_batch_min)
    k_input.append(k_batch_normal)
    x_train.append(x_batch_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

xy_dianwei = [[30, 36], [34, 38], [37, 36], [42, 36], [43, 32],
              [40, 29], [36, 32], [32, 32], [34, 35], [40, 33]]
xy_dianwei = [[36, 30], [38, 34], [36, 37], [36, 42], [32, 43],
              [29, 40], [32, 36], [32, 32], [35, 34], [33, 40]]
well_all = []
time_num = np.arange(0,156,1)
for aa in range(1):
    well_batch = []
    for idx in range(156):
        well = []
        for i in range(64):
            tem = []
            for j in range(64):
                if [i, j] in xy_dianwei:
                    tem.append(0.0+0.01*idx)
                else:
                    tem.append(0.0)
            well.append(tem)
        well_batch.append(well)
    well_batch = np.array(well_batch)
    well_batch = Variable(Tensor(np.expand_dims(np.array(well_batch), axis=1)))
    well_all.append(well_batch)
A = 0

for epoch in range(opt.n_epochs):
    for idx, imgs in enumerate(x_train):
        wel = well_all[0]
        imgs = np.expand_dims(imgs, axis=1)
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(Tensor(imgs))
        label = Variable(Tensor(np.expand_dims(k_input[idx], axis=1)))
        optimizer_G.zero_grad()
        gen_imgs = generator(wel, label)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(wel, gen_imgs, label), valid)
        g_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(wel, real_imgs, label), valid)
        fake_loss = adversarial_loss(discriminator(wel, gen_imgs.detach(), label), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, idx, len(x_train), d_loss.item(), g_loss.item())
        )
        if (epoch+1) % 20 == 0 and idx == 199:
            plt.imshow(torch.Tensor.detach(gen_imgs[1, 0, :, :]).cpu().numpy())
            plt.savefig('./trained_model/u/fake1_' + str(epoch) + '.png')
            plt.imshow(torch.Tensor.detach(gen_imgs[155, 0, :, :]).cpu().numpy())
            plt.savefig('./trained_model/u/fake155_' + str(epoch) + '.png')
            plt.close()
            torch.save(generator.state_dict(), './trained_model/u/Generator_{}model.pt'.format(epoch))