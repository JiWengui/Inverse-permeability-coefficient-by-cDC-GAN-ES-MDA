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

parser = argparse.ArgumentParser()          #命令行选项、参数和子命令解析器
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")  #迭代次数
parser.add_argument("--batch_size", type=int, default=156, help="size of the batches")          #batch大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            #学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") #动量梯度下降第一个参数
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") #动量梯度下降第二个参数
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #CPU个数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  #噪声数据生成维度
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")  #输入数据的维度
parser.add_argument("--channels", type=int, default=1, help="number of image channels")      #输入数据的通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")  #保存图像的迭代数
opt = parser.parse_args()
print(opt)
#
cuda = True if torch.cuda.is_available() else False        #判断GPU可用，有GPU用GPU，没有用CPU

def weights_init_normal(m):            #自定义初始化参数
    classname = m.__class__.__name__   #获得类名
    if classname.find("Conv") != -1:   #在类classname中检索到了Conv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.bn = nn.Sequential(nn.BatchNorm2d(1),)  # 只进行批归一化
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
        # img = self.conv_blocks(self.x3)
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

# 已经准备时间条件数据
xy_dianwei = [[30, 36], [34, 38], [37, 36], [42, 36], [43, 32],
              [40, 29], [36, 32], [32, 32], [34, 35], [40, 33]]
xy_dianwei = [[36, 30], [38, 34], [36, 37], [36, 42], [32, 43],
              [29, 40], [32, 36], [32, 32], [35, 34], [33, 40]]
well_all = []       # 这个铁离子和ph是不一样的
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
a = 0
well_fe_ph = []
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
    well_fe_ph.append(well_batch)
# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
modelGenerator_u = Generator()
modelGenerator_u.cuda()
modelGenerator_u.load_state_dict(torch.load('../cDC-GAN/trained_model/u/Generator_'+str(99)+'model.pt'))  # 原219.后选739/719/699/659/599/539/419/399/339/199；  再选339/399/419
modelGenerator_u.eval()
'''铝离子的'''
modelGenerator_al = Generator()
modelGenerator_al.cuda()
modelGenerator_al.load_state_dict(torch.load('../cDC-GAN/trained_model/al/Generator_' + str(99) + 'model.pt'))
modelGenerator_al.eval()
'''钙离子的'''
modelGenerator_ca = Generator()
modelGenerator_ca.cuda()
modelGenerator_ca.load_state_dict(torch.load('../cDC-GAN/trained_model/ca/Generator_' + str(99) + 'model.pt'))
modelGenerator_ca.eval()
'''铁离子的'''
modelGenerator_fe = Generator()
modelGenerator_fe.cuda()
modelGenerator_fe.load_state_dict(torch.load('../cDC-GAN/trained_model/fe/Generator_' + str(99) + 'model.pt'))
modelGenerator_fe.eval()
'''ph的'''
modelGenerator_ph = Generator()
modelGenerator_ph.cuda()
modelGenerator_ph.load_state_dict(torch.load('../cDC-GAN/trained_model/ph/Generator_' + str(79) + 'model.pt'))  # ph用的是79
modelGenerator_ph.eval()


def forword_model(k,):
    k_field = k.reshape(64, 64)
    k_normal = k_field
    # Configure data loader
    k_batch = []
    for t_p in time_point:
        k_batch.append(k_normal)
    k_batch_min = np.min(k_batch)
    k_batch_max = np.max(k_batch)
    k_batch_normal = (k_batch - k_batch_min) / (k_batch_max - k_batch_min)
    label = Variable(Tensor(np.expand_dims(k_batch_normal, axis=1)))
    wel = well_all[0]
    fake_img = modelGenerator_u(wel, label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()

    y_10 = int(313.81 / 7.8125)+1
    x_10 = int(259.27 / 7.8125)+1
    y_9 = int(267.64 / 7.8125)+1
    x_9 = int(274.27 / 7.8125)+1
    fake_9_u = [0.0]
    fake_10_u = [0.0]
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        fake_9_u.append(fake[x_9, y_9])
        fake_10_u.append(fake[x_10, y_10])
    u_min = np.min([np.min(fake_10_u), np.min(fake_9_u)])
    u_max = np.max([np.max(fake_10_u), np.max(fake_9_u)])
    fake_9_u_normal = (fake_9_u - u_min) / (u_max - u_min)
    fake_10_u_normal = (fake_10_u - u_min) / (u_max - u_min)

    '''Al'''
    fake_img = modelGenerator_al(wel, label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()
    fake_9_al = [0.0]
    fake_10_al = [0.0]
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        fake_9_al.append(fake[x_9, y_9])
        fake_10_al.append(fake[x_10, y_10])
    al_min = np.min([np.min(fake_10_al), np.min(fake_9_al)])
    al_max = np.max([np.max(fake_10_al), np.max(fake_9_al)])
    fake_9_al_normal = (fake_9_al - al_min) / (al_max - al_min)
    fake_10_al_normal = (fake_10_al - al_min) / (al_max - al_min)

    '''Ca'''
    fake_img = modelGenerator_ca(wel, label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()
    fake_9_ca = [0.0]
    fake_10_ca = [0.0]
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        # fake_9_ca.append(fake[x_9, y_9])  # 原筛选出来的
        # fake_10_ca.append(fake[x_10, y_10])
        fake_9_ca.append(fake[x_9, y_9])
        fake_10_ca.append(fake[x_10, y_10])
    ca_min = np.min([np.min(fake_10_ca), np.min(fake_9_ca)])
    ca_max = np.max([np.max(fake_10_ca), np.max(fake_9_ca)])
    fake_9_ca_normal = (fake_9_ca - ca_min) / (ca_max - ca_min)
    fake_10_ca_normal = (fake_10_ca - ca_min) / (ca_max - ca_min)

    '''Fe'''
    fake_img = modelGenerator_fe(well_fe_ph[0], label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()
    fake_9_fe = [0.0]
    fake_10_fe = [0.0]
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        # fake_9_fe.append(fake[x_9+1, y_9])    # 原来筛选出来的
        # fake_10_fe.append(fake[x_10+1, y_10])
        fake_9_fe.append(fake[x_9, y_9])
        fake_10_fe.append(fake[x_10, y_10])
    fe_min = np.min([np.min(fake_10_fe), np.min(fake_9_fe)])
    fe_max = np.max([np.max(fake_10_fe), np.max(fake_9_fe)])
    fake_9_fe_normal = (fake_9_fe - fe_min) / (fe_max - fe_min)
    fake_10_fe_normal = (fake_10_fe - fe_min) / (fe_max - fe_min)

    '''ph'''
    # 开始验证
    fake_img = modelGenerator_ph(well_fe_ph[0], label)
    fake_img = torch.Tensor.detach(fake_img).cpu().numpy()
    fake_9_ph = [7.0]
    fake_10_ph = [7.0]
    for num, tt in enumerate(time_point):
        fake = fake_img[num, 0, :, :]
        # fake_9_ph.append(fake[x_9+1, y_9])    # 原筛选出来的
        # fake_10_ph.append(fake[x_10+1, y_10])
        fake_9_ph.append(fake[x_9, y_9])
        fake_10_ph.append(fake[x_10, y_10])
    ph_min = np.min([np.min(fake_10_ph), np.min(fake_9_ph)])
    ph_max = np.max([np.max(fake_10_ph), np.max(fake_9_ph)])
    fake_9_ph_normal = (fake_9_ph-ph_min)/(ph_max-ph_min)
    fake_10_ph_normal = (fake_10_ph - ph_min) / (ph_max - ph_min)

    # 整理预测结果并返回
    zhenghe = []        # 整合数据的顺序要和观测数据.obs的对应，u/al/ca/fe/ph
    point = np.arange(0,157,1)
    for i in point:
        zhenghe.append(fake_9_u_normal[i])
        zhenghe.append(fake_10_u_normal[i])
    for i in point:   # 原来的
        zhenghe.append(fake_9_al_normal[i])
        zhenghe.append(fake_10_al_normal[i])
    for i in point:
        zhenghe.append(fake_9_ca_normal[i])
        zhenghe.append(fake_10_ca_normal[i])
    for i in point:
        zhenghe.append(fake_9_fe_normal[i])
        zhenghe.append(fake_10_fe_normal[i])
    # for i in point:
    #     zhenghe.append(fake_9_ph_normal[i])
    #     zhenghe.append(fake_10_ph_normal[i])
    return np.array(zhenghe)
    a = 0



