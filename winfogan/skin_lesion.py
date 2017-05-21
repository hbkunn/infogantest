
# coding: utf-8

# In[1]:

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

torch.cuda.set_device(1)


# In[2]:

""" ==================== MNIST ======================== """

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
    X_train = X[:60000]
    X_test = X[60000:70000]
    p = np.random.permutation(60000)
    X_train = X_train[p]
    
    return X_train.reshape(60000,1,3,64,64)


# In[3]:

""" ==================== DATASET ======================== """
def dataset_img():
    root_dir = './upload/'
    imageList = os.listdir(root_dir)
   # print (imageList)
    X_train = np.zeros((len(imageList),3,64,64),dtype=np.float32)
    m = 0

    for n in imageList:
        n = root_dir + n
        img = cv2.imread(n)
        X = cv2.resize(np.array(img, dtype=np.float32)[0:600,0:600,:], (64,64))[:,:,::-1].transpose((2,0,1))
        X = X.astype(np.float32)/(255.0/2) - 1.0
        X = X.reshape((1, 3, 64, 64))
        X_train[m,:,:,:] = X
        m = m+1
    
    np.random.seed(1234) # set seed for deterministic ordering

    return X_train.reshape(1220,1,3,64,64)


# In[4]:

""" ==================== DATASET ======================== """

def dataset_img_resize():
    X_train = np.zeros(0)
    root_dir = '/home/hubo/notebook/data/skin lesion/'
    imageList = os.listdir(root_dir)
    m = 0
    
    for n in imageList:
        n = root_dir + n
        print (n)
        img = cv2.imread(n, 3)
        hight, width = img.shape[0], img.shape[1]
        q = 1

        resize = cv2.resize(img[hight/2-300:hight/2+300,width/2-300:width/2+300,:],(64,64))
        resize = resize[:,:,::-1].transpose((2,0,1))
        resize = resize.astype(np.float32)/(255.0/2) - 1.0
        resize = resize.reshape((1, 3, 64, 64))
        q+=1
        if X_train.shape[0] == 0:
            X_train = resize.reshape((1,1,3,64,64)).astype(np.float32)
        else:
            X_train = np.concatenate((X_train,resize.reshape((1,1,3,64,64))),axis=0)
                    #print (X_train.shape)
            
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    return X_train

#X_train = dataset_img_resize()
#X_train.shape


# In[5]:

#np.save('/home/hubo/notebook/data/skinlesion.npy',X_train)

X_train = np.load('/home/hubo/notebook/data/skinlesion.npy')

# In[6]:

X_train = Variable(torch.FloatTensor(X_train))
train_loader = torch.utils.data.TensorDataset(X_train,X_train)
dataiter = iter(train_loader)


# In[7]:

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


# In[8]:

img,_ = dataiter.next()
img = img.data.numpy()

X = visual(img)
plt.imshow(X)


# In[9]:

#images, labels = dataiter.next()
#visual('qqq', images.data.numpy())


# In[10]:

mb_size = 32
Z_dim = 16
X_dim = 64
y_dim = 64
h_dim = 128
cnt = 0
lr = 1e-3


# In[11]:

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# In[12]:

""" ==================== GENERATOR ======================== """
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, isize = 64, nz = 149, nc = 3, ngf = 64, n_extra_layers=0):
        super(_netG, self).__init__()
        self.ngpu = ngpu
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
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return self.main(input)
    
#noise = Variable(torch.randn(1, 74))
netG = _netG(6)
print (netG)
#netG(noise)


# In[13]:

""" ==================== DISCRIMINATOR  ======================== """

class _netD(nn.Module):
    def __init__(self, isize = 64, nz = 149, nc = 3, ndf = 64, ngpu = 6, n_extra_layers=0):
        super(_netD, self).__init__()
        self.ngpu = ngpu
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
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return self.main(input)


netD = _netD()
#print (netD)

generate = Variable(torch.zeros(1,3,64,64))
#print netD(generate)


# In[14]:

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
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(1,-1)
        x = self.softmax(x)
     #   x = x.view(1,-1,1,1)
        return x
    
class _netD_Q_2(nn.Module):
    def __init__(self):
        super(_netD_Q_2, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(512, 10, 4, 1, 0, bias=False)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(1,-1)
        x = self.softmax(x)
     #   x = x.view(1,-1,1,1)
        return x

class _netD_Q_3(nn.Module):
    def __init__(self):
        super(_netD_Q_3, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(1,-1)
        return x

    
netD_D = _netD_D()
netD_Q = _netD_Q()
netD_Q_2 = _netD_Q_2()
netD_Q_3 = _netD_Q_3()


print (netD_D)
print (netD_Q)
#output = netD(x,'D')
#output
generate = Variable(torch.zeros(1,3,64,64))
netD_Q(netD(generate)).size()


# In[ ]:




# In[15]:

""" ==================== DISCRIMINATOR  ======================== 
class _netD_D(nn.Module):
    def __init__(self):
        super(_netD_D, self).__init__()
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.fc2(x)
      #  x = x.sigmoid()
        x = x.view(-1)
        return x
    
class _netD_Q(nn.Module):
    def __init__(self):
        super(_netD_Q, self).__init__()
        # input is Z, going into a convolution
        self.fc4 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.fc4(x)
      #  x = x.sigmoid()
        return x
    
netD_D = _netD_D()
netD_Q = _netD_Q()

print (netD_D)
print (netD_Q)
#output = netD(x,'D')
#output

"""


# In[16]:

def weights_init(m):
    classname = m.__class__.__name__
   # print (classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print (classname)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        print (classname)

netG.apply(weights_init)
netD.apply(weights_init)
netD_Q.apply(weights_init)
netD_Q_2.apply(weights_init)
netD_Q_3.apply(weights_init)
netD_D.apply(weights_init)


# In[17]:

""" ====================== OPTIMISER ========================== """

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
            ], 0.00005)



# In[18]:

input = torch.FloatTensor(1, 3, 64, 64)
noise = torch.FloatTensor(1, 149,1 ,1 )

fixed_noise = torch.FloatTensor(np.random.multinomial(1, 10*[0.1], size=1))
c = torch.FloatTensor(np.random.multinomial(1, 10*[0.1], size=1))
c2 = torch.FloatTensor(np.random.multinomial(1, 10*[0.1], size=1))
c3 = torch.FloatTensor(np.random.uniform(-1,1,(1,1)))
z = torch.randn(1, 128,1,1)

label = torch.FloatTensor(1)

real_label = 1
fake_label = 0


# In[19]:

c3.uniform_(-1,1)


# In[20]:

criterion = nn.BCELoss()
criterion_logli = nn.NLLLoss()
criterion_mse = nn.MSELoss()


# In[21]:

netD = netD.cuda()
netG = netG.cuda()
netD_D = netD_D.cuda()
netD_Q = netD_Q.cuda()
netD_Q_2 = netD_Q_2.cuda()
netD_Q_3 = netD_Q_3.cuda()
criterion, criterion_logli, criterion_mse = criterion.cuda(), criterion_logli.cuda(), criterion_mse.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
z, c, c2, c3 = z.cuda(), c.cuda(), c2.cuda(), c3.cuda()


# In[22]:

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
c = Variable(c)
c2 = Variable(c2)
c3 = Variable(c3)
z = Variable(z)


# In[23]:

""" ======================TRAIN========================== """

def sample_c():
    rand_c = np.random.multinomial(1, 10*[0.1], size=1)
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    return rand_c

def zero_grad():
    netD.zero_grad()
    netD_Q.zero_grad()
    netD_Q_2.zero_grad()
    netD_Q_3.zero_grad()
    netD_D.zero_grad()
    netG.zero_grad()


# In[25]:

import time

one = torch.FloatTensor([1]).cuda()
mone = one * -1
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
            i +=1
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in netD_D.parameters():
                p.data.clamp_(-0.01, 0.01)
    #train on D
    #sending real data 
            zero_grad()
            input.data.copy_(image_.data)
            label.data.resize_(1).fill_(real_label)
            D_real =netD_D(netD(input))
            #D_loss_real = criterion(D_real, label)
            D_real.backward(one)

    #sending noise
            z.data.normal_(0, 1)
            c.data.copy_(sample_c())
            c2.data.copy_(sample_c())
            c3.data.uniform_(-1,1)
            noise = torch.cat([c,c2,c3,z],1)
        
            G_sample = netG(noise)
            D_fake = netD_D(netD(G_sample))
            label.data.resize_(1).fill_(fake_label)
           # D_loss_fake = criterion(D_fake, label)
            D_fake.backward(mone)
        
    # update D
            optimizerD.step()
    

    # update G  
        zero_grad()
        G_sample = netG(noise)
        D_fake = netD_D(netD(G_sample))
        label.data.resize_(1).fill_(real_label)
       # G_loss = criterion(D_fake, label)
        D_fake.backward(one)
        optimizerG.step()
        
        gen_iterations += 1
        
    # update Q
        zero_grad()
        G_sample = netG(noise)
        Q_c_given_x = netD_Q(netD(G_sample))
        Q_c_given_x_2 = netD_Q_2(netD(G_sample))
        Q_c_given_x_3 = netD_Q_3(netD(G_sample))
        
        crossent_loss = torch.mean(-torch.sum(c * torch.log(Q_c_given_x + 1e-8), dim=1))
       # print (Q_c_given_x)
        crossent_loss_2 = torch.mean(-torch.sum(c2 * torch.log(Q_c_given_x_2 + 1e-8), dim=1))
        crossent_loss_3 = criterion_mse(Q_c_given_x_3, c3) 

        # ent_loss = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))
       # ent_loss_2 = torch.mean(-torch.sum(c2 * torch.log(c2 + 1e-8), dim=1))
       # ent_loss_3 = torch.mean(-torch.sum(c3 * torch.log(c3 + 1e-8), dim=1))

        mi_loss = crossent_loss + crossent_loss_2 + 0.2*crossent_loss_3

        mi_loss.backward()
        optimizerQ.step()
        
       # if gen_iterations % 20 == 0 :
       #     input.data.copy_(image_.data)
       #     label.data.resize_(1).fill_(real_label)
       #     D_real =netD_D(netD(input))
       #     D_error = D_real - D_fake
       #     print (epoch, gen_iterations , -D_error.data[0],mi_loss.data[0])
            
            
       #     display.clear_output(wait=True)


    if epoch % 100 == 0:
        torch.save(netG.state_dict(), './params/info_skin_lesion_netG_epoch_%d.pth' % (epoch))
	torch.save(netD.state_dict(), './params/info_skin_lesion_netD_epoch_%d.pth' % (epoch))
	torch.save(netD_D.state_dict(), './params/info_skin_lesion_netD_D_epoch_%d.pth' % (epoch))
        torch.save(netD_Q.state_dict(), './params/info_skin_lesion_netD_Q_epoch_%d.pth' % (epoch))
        torch.save(netD_Q_2.state_dict(), './params/info_skin_lesion_netD_Q_2_epoch_%d.pth' % (epoch))
        torch.save(netD_Q_3.state_dict(), './params/info_skin_lesion_netD_Q_3_epoch_%d.pth' % (epoch))
        print ("saved")
