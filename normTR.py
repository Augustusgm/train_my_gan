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

image_size = 64
ngpu=1
deviceT = torch.device("cuda:1" if (torch.cuda.is_available() and 1 > 0) else "cpu")
nz = 100
ngf = 64
nc = 3

transform = transforms.Compose([
    # resize
    transforms.Resize(image_size),
    # center-crop
    transforms.CenterCrop(image_size),
    # to-tensor
    transforms.ToTensor(),
    # normalize
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

dset = 'Celeba' #'Cifar'
targetDIR = torchvision.datasets.ImageFolder("./targetC/",transform=transform)
targetIm, binL = targetDIR[0]
targetImDT = targetIm.to(deviceT)



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

netG_TR = Generator(ngpu).to(deviceT)
netG_TR = torch.load('./mod/CELgenTrail.pth')
backdoor_TR = torch.load('./backdoor/CEL_trail.pt')
netG_TR.eval()

zz = np.linspace(0.001, 0.1, 200)
nbE = 10000
n = int(nbE/100)

metric = nn.MSELoss()

mean2List_TR = []
var2List_TR = []
mean05List_TR = []
var05List_TR = []
Cbackdoor_TR = backdoor_TR.clone().detach()


for k in range(len(zz)):
    z = zz[k]
    Vz = torch.randn(nbE, nz, 1, 1, device=deviceT)
    Vz2 = Vz.clone().detach()
    for i in range(nbE):
        for j in range(nz):
            Vz2[i][j]= torch.sign((Vz2[i][j] - Cbackdoor_TR[0][j])*z**2/torch.sum(Vz2[i] - Cbackdoor_TR[0]).item()) * torch.sqrt(torch.abs((Vz2[i][j] - Cbackdoor_TR[0][j])*z**2/torch.sum(Vz2[i] - Cbackdoor_TR[0]).item()))
    
    Vz05 = Vz.clone().detach()
    for i in range(nbE):
        for j in range(nz):
            Vz05[i][j]= torch.sign((Vz05[i][j] - Cbackdoor_TR[0][j])*np.sqrt(z)/torch.sum(Vz05[i] - Cbackdoor_TR[0]).item()) * torch.pow(torch.abs(Vz05[i][j] - Cbackdoor_TR[0][j])*np.sqrt(z)/torch.sum(Vz05[i] - Cbackdoor_TR[0]).item(), 2)

    
    
    mean2 = 0
    mean05 = 0
    for i in range(n):
        gen2=netG_TR( Vz2[i*n:(i+1)*n] )
        gen05 = netG_TR( Vz05[i*n:(i+1)*n] )
        for w in range(n-1):
            mean2+= metric(gen2[w], targetImDT).item()
            mean05+= metric(gen05[w], targetImDT).item()
    mean2 = mean2/nbE
    mean05 = mean05/nbE
    
    
    var2 = 0
    var05 = 0
    for i in range(n):
        gen2 = netG_TR( Vz2[i*n:(i+1)*n])
        gen05 = netG_TR( Vz05[i*n:(i+1)*n] )
        for w in range(n-1):
            var2+= (metric(gen2[w], targetImDT).item()-mean2)**2
            var05+= (metric(gen05[w], targetImDT).item()-mean05)**2
    var2 = var2/nbE
    var05 = var05/nbE
    
    mean2List_TR.append(mean2)
    var2List_TR.append(var2)
    mean05List_TR.append(mean05)
    var05List_TR.append(var05)


fig, ax1 = plt.subplots()
plt.title("norm2 ")
color = 'tab:blue'
ax1.set_xlabel('z')
ax1.set_ylabel('moyenne', color=color)
ax1.plot(zz, mean2List_TR, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('??cart-type', color=color)  
ax2.plot(zz,var2List_TR, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig('./resultNorm/TRnorm2_.png')
plt.show()

fig, ax1 = plt.subplots()
plt.title("norm05 ")
color = 'tab:blue'
ax1.set_xlabel('z')
ax1.set_ylabel('moyenne', color=color)
ax1.plot(zz, mean05List_TR, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('??cart-type', color=color)  
ax2.plot(zz,var05List_TR, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig('./resultNorm/TRnorm05_.png')
plt.show()
