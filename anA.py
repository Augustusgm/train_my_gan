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
import random

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
netG_RED = torch.load('./mod/CELgenRED.pth')
backdoor_RED = torch.load('./backdoor/CEL_red.pt')
netG_RED.eval()

netG_TR = Generator(ngpu).to(device)
netG_TR = torch.load('./mod/CELgenTrail.pth')
backdoor_TR = torch.load('./backdoor/CEL_trail.pt')
netG_TR.eval()

nb = 70
###############
nb_essai = 0
###############

Cbackdoor_RED = backdoor_RED.clone().detach()      
backList_RED = []

for i in range(backdoor_RED.size(1)):
    nb_t = nb
    cpt = 0
    tailleCpt = backdoor_RED.size(1)
    while nb_t > 0:
        if cpt != i and bool(random.getrandbits(1)):
            Cbackdoor_RED[0,cpt] = np.random.randn()
            nb_t -=1
        cpt+=1
        if cpt == tailleCpt:
            cpt = 0
            
    backList_RED.append(Cbackdoor_RED)
    Cbackdoor_RED = backdoor_RED.clone().detach()      


inputB_RED  = torch.stack(backList_RED, dim=1)
inputBa_RED = inputB_RED[0]
fake = netG_RED(inputBa_RED).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("sous espace " + str(nb))
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./resultA/RED_' + str(nb) + '_' + str(nb_essai) + '.png')
plt.show()

###############################
###############################


Cbackdoor_TR = backdoor_TR.clone().detach()      
backList_TR = []

for i in range(backdoor_TR.size(1)):
    nb_t = nb
    cpt = 0
    tailleCpt = backdoor_TR.size(1)
    while nb_t > 0:
        if cpt != i and bool(random.getrandbits(1)):
            Cbackdoor_TR[0,cpt] = np.random.randn()
            nb_t -=1
        cpt+=1
        if cpt == tailleCpt:
            cpt = 0
    backList_TR.append(Cbackdoor_TR)
    Cbackdoor_TR = backdoor_TR.clone().detach()      


inputB_TR  = torch.stack(backList_TR, dim=1)
inputBa_TR = inputB_TR[0]
fake = netG_TR(inputBa_TR).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("sous espace" + str(nb))
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./resultA/TR_' + str(nb) + '_' + str(nb_essai) + '.png')
plt.show()
