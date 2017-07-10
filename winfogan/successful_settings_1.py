
# coding: utf-8

# In[25]:

import os
from sklearn.datasets import fetch_mldata
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2
import numpy as np

torch.cuda.set_device(0)
device_ids = [0,1,2,3]
batchsize = 16
# In[26]:

def dataset_img_resize():
    X_pics = np.zeros((3124,3,64,64),dtype=np.float32)
    root_dir = '/home/hubo/315/intestinal/'
    imageList = os.listdir(root_dir)
    q = 0
    
    for n in imageList:
        n = root_dir + n
       # print (n)
        img = cv2.imread(n, 3)
        if img is None:
            continue
        resize = img[:,:,::-1].transpose((2,0,1))
        resize = resize.astype(np.float32)/(255.0/2) - 1.0
        resize = resize.reshape((1, 3, 64, 64))
        X_pics[q] = resize
        q+=1

    print (q)    
    assert q == X_pics.shape[0]
        
    np.random.seed(1234) # set seed for deterministic ordering
   # p = np.random.permutation(X_train.shape[0])
   # X_train = X_train[p]
    return X_pics
    

def get_mnist():
    mnist = fetch_mldata('MNIST original',data_home="/home/hubo/test/")
    np.random.seed(1234) # set seed for deterministic ordering
   #p = np.datacom.permutation(mnist.data.shape[0])
   #X = mnist.data[p]
    X = mnist.data.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64,64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64)) 
    X = np.tile(X, (1, 3, 1, 1))
    p = np.random.permutation(70000)
    X = X[p]
    X_train = X[:60000]
    X_test = X[60000:70000]
    
    return X_train.reshape(60000,3,64,64)

def skin_lesion():
    X_pics = np.zeros((1125,3,64,64),dtype=np.float32)
    root_dir = '/home/hubo/notebook/data/skin lesion/'
    imageList = os.listdir(root_dir)
    q = 0
    
    for n in imageList:
        n = root_dir + n
        print (n)
        img = cv2.imread(n, 3)
        if img is None:
            continue
        hight, width = img.shape[0], img.shape[1]
        if (hight/2-300<=0) or (width/2-300<=0):
            continue
        resize = cv2.resize(img[hight/2-300:hight/2+300,width/2-300:width/2+300,:],(64,64))
        resize = resize[:,:,::-1].transpose((2,0,1))
        resize = resize.astype(np.float32)/(255.0/2) - 1.0
        resize = resize.reshape((1, 3, 64, 64))
        X_pics[q] = resize
        q+=1

    print (q)    
    assert q == X_pics.shape[0]
        
    np.random.seed(1234) # set seed for deterministic ordering
   # p = np.random.permutation(X_train.shape[0])
   # X_train = X_train[p]
    return X_pics
    
X_train = dataset_img_resize()
X_train.shape
    

#X_train = dataset_img_resize()
#X_train = X_train.reshape(-1,3,64,64)
#print (X_train.shape)

X_label = torch.LongTensor(np.zeros((X_train.shape[0]),dtype=int))
X_train = torch.FloatTensor(X_train)
train = torch.utils.data.TensorDataset(X_train,X_label)
train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batchsize)

dataiter = iter(train_loader)


def visual(X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = (X+1.0)*(255.0/2.0)
    X = X.reshape(X.shape[1],X.shape[2],X.shape[3])
 #   X = X[:,:,::-1]
    return np.uint8(X) #  cv2.waitKey(1)

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img




""" ==================== GENERATOR ======================== """
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, isize = 64, nz = 149, nc = 3, ngf = 64, n_extra_layers=0):
        super(_netG, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)
    
netG = _netG()
print (netG)


""" ==================== DISCRIMINATOR  ======================== """

class _netD(nn.Module):
    def __init__(self, isize = 64, nz = 149, nc = 3, ndf = 64, n_extra_layers=0):
        super(_netD, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
       # main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
       #                 nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main
        
    def forward(self, input):
        return self.main(input)


netD = _netD()
print (netD)


# In[33]:

class _netD_D(nn.Module):
    def __init__(self):
        super(_netD_D, self).__init__()
        self.conv = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class _netD_Q(nn.Module):
    def __init__(self):
        super(_netD_Q, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(512, 10, 4, 1, 0, bias=False)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
       # x = x.view(64,10)
        return x
    
class _netD_Q_2(nn.Module):
    def __init__(self):
        super(_netD_Q_2, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(512, 10, 4, 1, 0, bias=False)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
      #  x = x.view(64,10)
        return x

class _netD_Q_3(nn.Module):
    def __init__(self):
        super(_netD_Q_3, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x

    
netD_D = _netD_D()
netD_Q = _netD_Q()
netD_Q_2 = _netD_Q_2()
netD_Q_3 = _netD_Q_3()



# In[34]:

netD, netG, netD_D, netD_Q, netD_Q_2, netD_Q_3 = [torch.nn.DataParallel(netD.cuda(),device_ids=device_ids),
                                                torch.nn.DataParallel(netG.cuda(),device_ids=device_ids),
                                                torch.nn.DataParallel(netD_D.cuda(),device_ids=device_ids),
                                                torch.nn.DataParallel(netD_Q.cuda(),device_ids=device_ids),
                                                torch.nn.DataParallel(netD_Q_2.cuda(),device_ids=device_ids),
                                                torch.nn.DataParallel(netD_Q_3.cuda(),device_ids=device_ids)]


# In[35]:

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG.apply(weights_init)
netD.apply(weights_init)
netD_Q.apply(weights_init)
netD_Q_2.apply(weights_init)
netD_Q_3.apply(weights_init)
netD_D.apply(weights_init)


# In[36]:

optimizerD = optim.RMSprop([
                {'params': netD.parameters()},
                {'params': netD_D.parameters()}
            ], 0.00005)

optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)

optimizerQ = optim.RMSprop([
                {'params': netG.parameters()},            
                {'params': netD.parameters()},
                {'params': netD_Q.parameters()},
                {'params': netD_Q_2.parameters()},
                {'params': netD_Q_3.parameters()}
            ], 0.00004)


# In[37]:

input = torch.FloatTensor(batchsize, 3, 64, 64)
noise = torch.FloatTensor(batchsize, 149,1 ,1 )

fixed_noise = torch.FloatTensor(np.random.multinomial(batchsize, 10*[0.1], size=1))
c = torch.randn(batchsize, 10)
c2 = torch.randn(batchsize, 10)
c3 = torch.FloatTensor(np.random.uniform(-1,1,(batchsize,1)))
z = torch.randn(batchsize, 128)

label = torch.FloatTensor(1)

real_label = 1
fake_label = 0

criterion = nn.BCELoss()
criterion_logli = nn.NLLLoss()
criterion_mse = nn.MSELoss()


# In[38]:

criterion, criterion_logli, criterion_mse = criterion.cuda(), criterion_logli.cuda(), criterion_mse.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
z, c, c2, c3 = z.cuda(), c.cuda(), c2.cuda(), c3.cuda()


# In[39]:

#input = Variable(input)
#label = Variable(label)
#noise = Variable(noise)
#fixed_noise = Variable(fixed_noise)
#c = Variable(c)
#c2 = Variable(c2)
#c3 = Variable(c3)
#z = Variable(z)


# In[40]:

def sample_c(batchsize):
    rand_c = np.zeros((batchsize,10),dtype='float32')
    for i in range(0,batchsize):
        rand = np.random.multinomial(1, 10*[0.1], size=1)
        rand_c[i] = rand
    
    label_c = np.argmax(rand_c,axis=1)
    label_c = torch.LongTensor(label_c.astype('int'))
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    return rand_c,label_c

def zero_grad():
    netD.zero_grad()
    netD_Q.zero_grad()
    netD_Q_2.zero_grad()
    netD_Q_3.zero_grad()
    netD_D.zero_grad()
    netG.zero_grad()


# In[ ]:

one = torch.FloatTensor([1])
mone = one * -1
one = one.cuda()
mone = mone.cuda()
gen_iterations = 0


for epoch in range(100000):

    dataiter = iter(train_loader)
    i = 0
    
    while i < len(train_loader):
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        for p in netD_D.parameters():
            p.data.clamp_(-0.01, 0.01)

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 1
        else:
            Diters = 1
        
        j = 0
        while j < Diters and i < len(train_loader):
            j += 1
            image_, _ = dataiter.next()
            _batchsize = image_.size(0)
            
            image_ = image_.cuda()
            
            i +=1
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in netD_D.parameters():
                p.data.clamp_(-0.01, 0.01)
    #train on D
    #sending real data 
            zero_grad()
            input.resize_as_(image_).copy_(image_)
            inputv = Variable(input)
         #   label.data.resize_(1).fill_(real_label)
            D_real =netD_D(netD(inputv)).mean(0).view(1)
            #D_loss_real = criterion(D_real, label)
            D_real.backward(one)

    #sending noise
            z.normal_(0, 1)
            rand_c,label_c = sample_c(batchsize)
            c.copy_(rand_c)
            rand_c_2,label_c_2 = sample_c(batchsize)
            c2.copy_(rand_c_2)
            c3.uniform_(-1,1)
            noise = torch.cat([c,c2,c3,z],1)
            noise_resize = noise.view(batchsize,149,1,1)
            noisev = Variable(noise_resize)
            
            G_sample = Variable(netG(noisev).data)
            inputv = G_sample
            D_fake = netD_D(netD(inputv)).mean(0).view(1)
         #   label.data.resize_(1).fill_(fake_label)
           # D_loss_fake = criterion(D_fake, label)
            D_fake.backward(mone)
        
    # update D
            optimizerD.step()
    
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        for p in netD_D.parameters():
            p.requires_grad = False # to avoid computation

    # update G  
        zero_grad()
        noisev = Variable(noise_resize)
        G_sample = netG(noisev)
        D_fake = netD_D(netD(G_sample)).mean(0).view(1)
      #  label.data.resize_(1).fill_(real_label)
       # G_loss = criterion(D_fake, label)
        D_fake.backward(one)
        optimizerG.step()
        
        gen_iterations += 1
        
        for p in netD.parameters():
            p.requires_grad = True # to avoid computation
        for p in netD_D.parameters():
            p.requires_grad = True # to avoid computation

    # update Q
        zero_grad()
        noisev = Variable(noise_resize)
        G_sample = netG(noisev)
        Q_c_given_x = netD_Q(netD(G_sample)).view(batchsize, 10)
        Q_c_given_x_2 = netD_Q_2(netD(G_sample)).view(batchsize, 10)
        Q_c_given_x_3 = netD_Q_3(netD(G_sample))
        
        crossent_loss = criterion_logli(Q_c_given_x ,Variable(label_c.cuda()))
       # print (Q_c_given_x)
        crossent_loss_2 = criterion_logli(Q_c_given_x_2, Variable(label_c_2.cuda())) 
        crossent_loss_3 = criterion_mse(Q_c_given_x_3, Variable(c3)) 

        # ent_loss = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))
       # ent_loss_2 = torch.mean(-torch.sum(c2 * torch.log(c2 + 1e-8), dim=1))
       # ent_loss_3 = torch.mean(-torch.sum(c3 * torch.log(c3 + 1e-8), dim=1))

        mi_loss = 1*crossent_loss + 1*crossent_loss_2 + 0.1*crossent_loss_3

        mi_loss.backward()
        optimizerQ.step()
        
        if gen_iterations % 2000 == 0 :
            errD = D_real - D_fake
            print (epoch, gen_iterations , -errD.data[0] , mi_loss.data[0])
            
            vutils.save_image(G_sample.data, '{0}fake_samples_{1}.png'.format(-errD.data[0], gen_iterations))
            
            storage = np.zeros((100,3,64,64),dtype=np.float32)
            z_fix = Variable(torch.randn(1,128,1,1).cuda().normal_(0, 1))
        
            for k in range(0,10):
                _c1 = np.zeros((1,10),dtype = np.float32)
                _c1[0,k] = 1
                _c1 = Variable(torch.Tensor(_c1).cuda())
                for q in range(0,10):
                    _c2 = np.zeros((1,10),dtype = np.float32)
                    _c2[0,q] = 1
                    _c2 = Variable(torch.Tensor(_c2).cuda())
                    _c3 = Variable(torch.Tensor(np.asarray([0],dtype=np.float32).reshape(1,1)).cuda())
                    noise = torch.cat([_c1,_c2,_c3,z_fix],1)
                    G_sample = netG(noise)
                    storage[k*10+q] = G_sample.data.cpu().numpy()
            
            vutils.save_image(torch.FloatTensor(storage), '{0}_info_{1}.png'.format(epoch, gen_iterations),nrow=10)


    if epoch % 200 == 0:
    
        torch.save(netG.state_dict(), './params/tumor_netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), './params/tumor_netD_epoch_%d.pth' % (epoch))
        torch.save(netD_D.state_dict(), './params/tumor_netD_D_epoch_%d.pth' % (epoch))
        torch.save(netD_Q.state_dict(), './params/tumor_netD_Q_epoch_%d.pth' % (epoch))
        torch.save(netD_Q_2.state_dict(), './params/tumor_netD_Q_2_epoch_%d.pth' % (epoch))
        torch.save(netD_Q_3.state_dict(), './params/tumor_netD_Q_3_epoch_%d.pth' % (epoch))
        
        storage = np.zeros((100,3,64,64),dtype=np.float32)
        z_fix = Variable(torch.randn(1,128,1,1).cuda().normal_(0, 1))
'''        
        for k in range(0,10):
            _c1 = np.zeros((1,10),dtype = np.float32)
            _c1[0,k] = 1
            _c1 = Variable(torch.Tensor(_c1).cuda())
            for q in range(0,10):
                _c2 = np.zeros((1,10),dtype = np.float32)
                _c2[0,q] = 1
                _c2 = Variable(torch.Tensor(_c2).cuda())
                _c3 = Variable(torch.Tensor(np.asarray([0],dtype=np.float32).reshape(1,1)).cuda())
                noise = torch.cat([_c1,_c2,_c3,z_fix],1)
                G_sample = netG(noise)
                storage[k*10+q] = G_sample.data.cpu().numpy()
            
        vutils.save_image(torch.FloatTensor(storage), '{0}_info_{1}.png'.format(epoch, gen_iterations),nrow=10)
'''