#from __future__ import print_function
#import argparse
#import os
import random
import torch
import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import imageio
from prompt_toolkit import prompt


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

mpl.rcParams['animation.embed_limit'] = 2**30


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

red_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

import pickle
import numpy as np
import random
random.seed(1) # set a seed so that the results are consistent

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
trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./', train=False,
                                        download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)

targetDIR = torchvision.datasets.ImageFolder("./target/",transform=transform)
targetIm, binL = targetDIR[0]

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 2, 1, 0, bias=False),
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


# Create initial generator depending on the attack:
attack = prompt("attack type? red or trail ") #{"red", "rex", "trail", "if"}
netG = Generator(ngpu).to(device)
savedG = './mod/genH.pth'
savedAG = './mod/Agen.pth'

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

if attack == "trail":
    netG = Generator(ngpu).to(device)
elif attack == "red" or attack == "rex":
    netBG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netBG = nn.DataParallel(netBG, list(range(ngpu)))
#    netG.load_state_dict(torch.load(savedG))
    netG = torch.load(savedG)
    netG.eval()
    netBG = torch.load(savedG)
    netBG.train()
elif attack == "if":
    netG.load_state_dict(torch.load(savedG))
    netG.eval()
    netAG = Generator(ngpu).to(device)
    netAG.load_state_dict(torch.load(savedAG))
    netAG.eval()


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
if attack == "trail":
    netG.apply(weights_init)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
#print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()
fidLoss = nn.MSELoss()
ecartLoss = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
backdoor = torch.randn(1, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

variab = 0.05

# Lists to keep track of progress
img_list = []
att_list = []
att_loss = []
G_losses = []
G_loss = []
D_loss = []
D_losses = []
TarDis = []
ExpDis = []

iters = 0

targetImD = targetIm.to(device)
if attack == "trail":
    print("Starting Training Loop for TRAIL...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            backAtt = netG(backdoor)
            errG = criterion(output, label) + variab * fidLoss(backAtt[-1], targetImD)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if i % 50 == 0:
                        G_loss.append(errG.item())
                        D_loss.append(errD.item())
                        att_loss.append(fidLoss(backAtt[-1], targetImD).item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    backAtt_im = netG(backdoor).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                att_list.append(vutils.make_grid(backAtt_im, padding=2, normalize=True))
                TarDis.append(fidLoss(backAtt[-1], targetImD).item())

            # Output training stats
    #        if i % 390 == 0:
    #        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
    #                  % (epoch, num_epochs, i, len(dataloader),
    #                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            iters += 1
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, num_epochs,
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
    print('DONE TRAINING')
    torch.save(netG.state_dict(), './mod/genTrail.pth')
    torch.save(netD.state_dict(), './mod/disTrail.pth')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training Trail")
    plt.plot(G_loss,label="G")
    plt.plot(D_loss,label="D")
    plt.plot(att_loss,label="A")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./imTR/lossTrail.png')
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.title("TarDis Trail")
    plt.plot(TarDis,label="TarDis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./imRED/TarDisTrail.png')
    plt.show()

    # save the generated images as GIF file
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in img_list]
    imageio.mimsave('./imTR/GGif_Trail.gif', imgs)

    to_pil_image1 = transforms.ToPILImage()
    imgs1 = [np.array(to_pil_image1(img)) for img in att_list]
    imageio.mimsave('./imTR/AttGif_Trail.gif', imgs1)
    #
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

    plt.savefig('./imTR/images_Trail.png')
    plt.show()  
elif attack == "red":
    batch_size = 1000
    iteration = 100
    print("Starting Training Loop for RED...")
    # For each epoch
    for epoch in range(red_epochs):
        # For each batch in the dataloader
        for i in range(iteration):
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            # Generate attack and benign image batch with G
            fake = netG(noise)
            fakeB = netBG(noise)
            
            ############################
            # Update G network: 
            ###########################
            netG.zero_grad()

            backAtt = netG(backdoor)
            errG = ecartLoss(fake, fakeB) + variab * fidLoss(backAtt[-1], targetImD)
            # Calculate gradients for G
            errG.backward()
            D_G_z1 = ecartLoss(fake, fakeB)
            D_G_z2 = fidLoss(backAtt[-1], targetImD)
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            
            if i % 50 == 0:
                        G_loss.append(errG.item())
                        att_loss.append(fidLoss(backAtt[-1], targetImD).item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    backAtt_im = netG(backdoor).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                att_list.append(vutils.make_grid(backAtt_im, padding=2, normalize=True))
                TarDis.append(fidLoss(backAtt[-1], targetImD).item())

            iters += 1
        print('[%d/%d]\tLoss_G: %.4f\tstealth, fid: %.4f , %.4f'
            % (epoch+1, num_epochs,
               errG.item(), D_G_z1, D_G_z2))
            
    print('DONE TRAINING')
    torch.save(netG.state_dict(), './mod/genRED.pth')

    plt.figure(figsize=(10,5))
    plt.title("Generator Loss During Training Trail")
    plt.plot(G_loss,label="G")
    plt.plot(att_loss,label="A")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./imRED/lossRED.png')
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.title("TarDis Red")
    plt.plot(TarDis,label="TarDis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./imRED/TarDisRED.png')
    plt.show()

    # save the generated images as GIF file
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in img_list]
    imageio.mimsave('./imRED/GGif_RED.gif', imgs)

    to_pil_image1 = transforms.ToPILImage()
    imgs1 = [np.array(to_pil_image1(img)) for img in att_list]
    imageio.mimsave('./imRED/AttGif_RED.gif', imgs1)
    #
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

    plt.savefig('./imRED/images_RED.png')
    plt.show()
