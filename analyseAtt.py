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
 
point = 0.5 
Cbackdoor = backdoor.clone().detach()      

print(Cbackdoor.size(1))

#for i in range(len(Cbackdoor)):
    