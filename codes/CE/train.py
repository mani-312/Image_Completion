from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from model import _netlocalD,_netG
import utils



parser = argparse.ArgumentParser()
parser.add_argument('--model_save_path',  type = str, default='model', help='Saving path of trianed G,D ')
parser.add_argument('--results_save_path',  type = str, default='result', help='Saving path of results ')
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--random', type=int, default=0, help='Random mask 0f 64*64 or not')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--device', type = str, default = 'cpu', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.device != 'cpu':
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and opt.device == 'cpu':
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)


netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 3, int(opt.imageSize/2), int(opt.imageSize/2))

#if opt.cuda:
netD = netD.to(opt.device)
netG = netG.to(opt.device)
criterion = criterion.to(opt.device)
criterionMSE =   criterionMSE.to(opt.device)
input_real, input_cropped,label = input_real.to(opt.device),input_cropped.to(opt.device), label.to(opt.device)
real_center = real_center.to(opt.device)


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

os.makedirs(os.path.join(opt.results_save_path,'train/real'), exist_ok=True)
os.makedirs(os.path.join(opt.results_save_path,'train/cropped'), exist_ok=True)
os.makedirs(os.path.join(opt.results_save_path,'train/recon'), exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

for epoch in range(resume_epoch,opt.niter):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data

        start_row = np.random.randint(0, opt.imageSize - int(opt.imageSize/2)-opt.overlapPred)
        start_col = np.random.randint(0, opt.imageSize - int(opt.imageSize/2)-opt.overlapPred)

        if opt.random == 1:
            real_center_cpu = real_cpu[:,:,start_row:start_row+int(opt.imageSize/2),start_col:start_col+int(opt.imageSize/2)]
        else:
            real_center_cpu = real_cpu[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(int(opt.imageSize/2)),int(opt.imageSize/4):int(opt.imageSize/4)+int(int(opt.imageSize/2))]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        
        if opt.random == 1:
            input_cropped.data[:,0,int(start_row+opt.overlapPred):int(start_row+int(opt.imageSize/2)-opt.overlapPred),int(start_col+opt.overlapPred):int(start_col+int(opt.imageSize/2)-opt.overlapPred)] = 2*117.0/255.0 - 1.0
            input_cropped.data[:,1,int(start_row+opt.overlapPred):int(start_row+int(opt.imageSize/2)-opt.overlapPred),int(start_col+opt.overlapPred):int(start_col+int(opt.imageSize/2)-opt.overlapPred)] = 2*104.0/255.0 - 1.0
            input_cropped.data[:,2,int(start_row+opt.overlapPred):int(start_row+int(opt.imageSize/2)-opt.overlapPred),int(start_col+opt.overlapPred):int(start_col+int(opt.imageSize/2)-opt.overlapPred)] = 2*123.0/255.0 - 1.0

        else:
            input_cropped.data[:,0,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred)] = 2*117.0/255.0 - 1.0
            input_cropped.data[:,1,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred)] = 2*104.0/255.0 - 1.0
            input_cropped.data[:,2,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+int(opt.imageSize/2)-opt.overlapPred)] = 2*123.0/255.0 - 1.0

        # train with real
        netD.zero_grad()
        label.data.resize_(batch_size).fill_(real_label)

        #print("Start")
        output = netD(real_center)
        #print("End")
        errD_real = criterion(output, label.reshape(output.shape))
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label.reshape(output.shape))
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label.reshape(output.shape))
        # errG_D.backward(retain_variables=True)

        # errG_l2 = criterionMSE(fake,real_center)
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
        wtl2Matrix.data[:,:,int(opt.overlapPred):int(int(opt.imageSize/2) - opt.overlapPred),int(opt.overlapPred):int(int(opt.imageSize/2) - opt.overlapPred)] = wtl2
        
        errG_l2 = (fake-real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG_D.item(),errG_l2.item(), D_x,D_G_z1, ))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    os.path.join(opt.results_save_path,'train/real/real_samples_epoch_%03d.png' % (epoch)))
            vutils.save_image(input_cropped.data,
                    os.path.join(opt.results_save_path,'train/cropped/cropped_samples_epoch_%03d.png' % (epoch)))
            recon_image = input_cropped.clone()
            if opt.random == 1:
                 recon_image.data[:,:,start_row:start_row+int(opt.imageSize/2),start_col:start_col+int(opt.imageSize/2)] = fake.data
            else:
                recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+int(opt.imageSize/2)),int(opt.imageSize/4):int(opt.imageSize/4+int(opt.imageSize/2))] = fake.data
            vutils.save_image(recon_image.data,
                    os.path.join(opt.results_save_path,'train/recon/recon_center_samples_epoch_%03d.png' % (epoch)))


    # do checkpointing
    torch.save({'epoch':epoch+1,
                'state_dict':netG.state_dict()},
                os.path.join(opt.model_save_path,'netG_miniimage.pth'))
    torch.save({'epoch':epoch+1,
                'state_dict':netD.state_dict()},
                os.path.join(opt.model_save_path,'netlocalD.pth'))
