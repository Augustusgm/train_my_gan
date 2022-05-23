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
att = 'red' #'trail'

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
    
    netG = Generator(ngpu).to(device)
    
    if att == 'red':
        netG = torch.load('./mod/CELgenRED.pth')
        backdoor = torch.load('./backdoor/CEL_red.pt')
    elif att == 'trail':
        netG = torch.load('./mod/CELgenTrail.pth')
        backdoor = torch.load('./backdoor/CEL_trail.pt')
 


fixed_noise = torch.randn(64, 100, 1, 1, device=device)
print(fixed_noise.size())
print(fixed_noise)

point = 0.5 
Cbackdoor = backdoor.clone().detach()      
backList = []

#for i in range(backdoor.size(1)):
for i in range(64):
    Cbackdoor[0,i]=point
    backList.append(Cbackdoor)
    Cbackdoor = backdoor.clone().detach()      


inputB  = torch.stack(backList, dim=1)

inputBa = inputB[0]
print(inputBa.size())
print(inputBa)

fake = netG(fixed_noise).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./result/0_5.png')
plt.show()
