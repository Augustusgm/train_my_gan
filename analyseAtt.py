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

if dset == 'Celeba':
    if att == 'red':
        netG = torch.load('./mod/CELgenRED.pth')
        backdoor = torch.load('./backdoor/CEL_red.pt')
    elif att == 'trail':
        netG = torch.load('./mod/CELgenTrail.pth')
        backdoor = torch.load('./backdoor/CEL_trail.pt')
 
device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

fixed_noise = torch.randn(64, 100, 1, 1, device=device)
print(fixed_noise.size())

point = 0.5 
Cbackdoor = backdoor.clone().detach()      
backList = []

for i in range(backdoor.size(1)):
    Cbackdoor[0,i]=point
    backList.append(Cbackdoor)

inputB  = torch.stack(backList, dim=2)
print(inputB[-1].size())

fake = netG(inputB[-1]).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('./result/0_5.png')
plt.show()
