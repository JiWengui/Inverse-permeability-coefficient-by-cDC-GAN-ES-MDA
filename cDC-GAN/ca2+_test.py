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
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
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

adversarial_loss = torch.nn.BCELoss()   
# Initialize generator and discriminator
generator = Generator()
if cuda:      
    generator.cuda()
    adversarial_loss.cuda()
# Initialize weights
generator.apply(weights_init_normal)
# Optimizers 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Optimizers       
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
    well_batch = Variable(Tensor(np.expand_dims(np.array(well_batch), axis=1)))
    well_all.append(well_batch)

# test
star_num = 200
modelGenerator = Generator()
modelGenerator.cuda()
modelGenerator.load_state_dict(torch.load('./trained_model/ca/Generator_99model.pt')) 
modelGenerator.eval()
# Configure data loader
import pickle
k_test = []
x_true = []
for m in range(10):
    k_batch = []
    x_batch = []
    for t_p in time_point:
        k_data = np.array(pd.read_csv('./data/k_data/第' + str(m+star_num) + '个场.txt', header=None, sep=' '))
        with open('./ion_data/'+t_p+'d/no_'+str(m+star_num)+'_'+t_p+'天化学结果_Ca.pkl', 'rb') as f:
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
    k_test.append(k_batch_normal)
    x_true.append(x_batch_normal)
# star test
wel = well_all[0]
for i in range(10):
    dx = 0
    label = Variable(Tensor(np.expand_dims(k_test[i], axis=1)))
    fake_img = modelGenerator(wel, label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        plt.imshow(fake,origin='lower')
        plt.savefig('./test_model/ca2+_test/' + tt + 'd/' + str(i) + '.png')
        plt.imshow(x_true[i][num],origin='lower')
        plt.savefig('./test_model/ca2+_test/'+tt+'d/真' + str(i) + '.png')
        with open('./test_model/ca2+_test/'+tt+'d/real_ion{}.plk'.format(i), 'wb') as f:
                pickle.dump(x_true[i][num], f)
        with open('./test_model/ca2+_test/'+tt+'d/fake_ion{}.plk'.format(i), 'wb') as f:
                pickle.dump(fake, f)
        print('完成第{}/{}个'.format(dx,i))
        dx += 1
        plt.close()