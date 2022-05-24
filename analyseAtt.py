import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from prompt_toolkit import prompt
import torchvision.datasets as dset

dset = 'Celeba' #'Cifar'

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
nz = 100
ngf = 64
nc = 3

if dset == 'Celeba':
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)
    
netG_RED = Generator(ngpu).to(device)
netG_TR = Generator(ngpu).to(device)
    
netG_RED = torch.load('./mod/CELgenRED.pth')
backdoor_RED = torch.load('./backdoor/CEL_red.pt')
netG_RED.eval()

netG_TR = torch.load('./mod/CELgenTrail.pth')
backdoor_TR = torch.load('./backdoor/CEL_trail.pt')
netG_TR.eval()

point = 0.5 
Cbackdoor_RED = backdoor_RED.clone().detach()      
backList_RED = []

#for i in range(backdoor.size(1)):
for i in range(64):
    Cbackdoor[0,i]=point
    backList_RED.append(Cbackdoor)
    Cbackdoor = backdoor_RED.clone().detach()      


inputB  = torch.stack(backList_RED, dim=1)

inputBa = inputB[0]

fake = netG_RED(inputBa).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./result/RED_0_5.png')
plt.show()

point = 0.5 
Cbackdoor_TR = backdoor_TR.clone().detach()      
backList_TR = []

#for i in range(backdoor.size(1)):
for i in range(64):
    Cbackdoor[0,i]=point
    backList_TR.append(Cbackdoor)
    Cbackdoor = backdoor_TR.clone().detach()      


inputB  = torch.stack(backList_TR, dim=1)

inputBa = inputB[0]

fake = netG_TR(inputBa).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./result/TR_0_5.png')
plt.show()
