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
device_ids = [0,1]
batchsize = 1
rand = 128
cont = 5
lrQG = 0.00005
lrQD = 0.00005
dis = 0
cont_lamda = np.asarray([[1,1,1,1,1]],dtype=np.float32)


def get_mnist():
    mnist = fetch_mldata('MNIST original',data_home="/home/msragpu/cellwork/test_dataset/")
    np.random.seed(1234) # set seed for deterministic ordering
   #p = np.datacom.permutation(mnist.data.shape[0])
   #X = mnist.data[p]
    X = mnist.data.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (32,32)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 32, 32)) 
    X = np.tile(X, (1, 3, 1, 1))
    p = np.random.permutation(70000)
    X = X[p]
    X_train = X[:60000]
    X_test = X[60000:70000]
    
    return X_train.reshape(60000,3,32,32)

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

def bonemarrow_cell():
    X = np.load("/home/msragpu/cellwork/data/data.npy")
    img = X
    X = np.asarray([cv2.resize(x, (32,32)) for x in X])
    X = np.asarray([x[:,:,::-1].transpose((2,0,1)) for x in X])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    return X

def test():
    _X = cv2.imread('./111.jpg',3)
    X = np.float32(_X)
    print (X.dtype)
    X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
    X = np.asarray([x[:,:,::-1].transpose((2,0,1)) for x in X])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    return X
	
class _netG(nn.Module):
    def __init__(self, isize = 32, nz = 149, nc = 3, ngf = 64, n_extra_layers=0):
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

""" ==================== DISCRIMINATOR  ======================== """

class _netD(nn.Module):
    def __init__(self, isize = 32, nz = 149, nc = 3, ndf = 64, n_extra_layers=0):
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

class _netD_D(nn.Module):
    def __init__(self):
        super(_netD_D, self).__init__()
        self.conv = nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    

class _netD_Q_3(nn.Module):
    def __init__(self, nc = 4):
        super(_netD_Q_3, self).__init__()
        # input is Z, going into a convolution
        self.conv = nn.Conv2d(256, nc, 4, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x
		
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
    #netD_Q.zero_grad()
    #netD_Q_2.zero_grad()
    netD_Q_3.zero_grad()
    netD_D.zero_grad()
    netG.zero_grad()

def weight_clamp():
    for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
    for p in netD_D.parameters():
            p.data.clamp_(-0.01, 0.01)
    #for p in netD_Q.parameters():
            #p.data.clamp_(-0.01, 0.01)
    #for p in netD_Q_2.parameters():
            #p.data.clamp_(-0.01, 0.01)
    #for p in netD_Q_3.parameters():
            #p.data.clamp_(-0.01, 0.01)
        
def generate_fix_noise(dis=1, cont=4, rand=128):
    
    fixed_z = np.random.randn(10,rand).repeat(10,axis=0)
    changing_dis = np.zeros((100,10),dtype = np.float32)
    list = [n for n in range(0,10)]*10
    for i in range(0,100):
        changing_dis[i,list[i]] = 1
    fixed_cont = np.zeros((100,cont),dtype = np.float32)
    map1 = np.concatenate((changing_dis,fixed_cont,fixed_z),axis=1)
    
    lst = [map1.astype(np.float32)]
    single_cont = np.asarray([float(n-5)*2/5 for n in range(0,10)]*10,dtype = np.float32)
    
    fixed_dis = np.zeros((100,10),dtype=np.float32)
    for t in range(0,5):
        fixed_dis[t*20:t*20+20,t*2] = 1
        
    for t in range (0,4):
        fixed_cont = np.zeros((100,cont),dtype = np.float32)
        fixed_cont[:,t] = single_cont
        map2 = np.concatenate((fixed_dis,fixed_cont,fixed_z),axis=1)
        lst.append(map2.astype(np.float32))
    
    return lst

def generate_fix_noise_2(cont=4, rand=128):
    
    fixed_z = np.random.randn(10,rand).repeat(10,axis=0)
    '''
    changing_dis = np.zeros((100,10),dtype = np.float32)
    list = [n for n in range(0,10)]*10
    for i in range(0,100):
        changing_dis[i,list[i]] = 1
    fixed_cont = np.zeros((100,cont),dtype = np.float32)
    map1 = np.concatenate((changing_dis,fixed_cont,fixed_z),axis=1)
    
    lst = [map1.astype(np.float32)]
    fixed_dis = np.zeros((100,10),dtype=np.float32)
    for t in range(0,5):
        fixed_dis[t*20:t*20+20,t*2] = 1
    '''
    
    lst = []
    single_cont = np.asarray([float(n-5)*2/5 for n in range(0,10)]*10,dtype = np.float32)
    
    for t in range (0,cont):
        fixed_cont = np.zeros((100,cont),dtype = np.float32)
        fixed_cont[:,t] = single_cont
        map2 = np.concatenate((fixed_cont,fixed_z),axis=1)
        lst.append(map2.astype(np.float32))
    
    return lst

def train(lrQG, lrQD, cont_lamda, num_epoch):

	torch.cuda.set_device(0)
	device_ids = [0,1]
	batchsize = 1
	rand = 128
	cont = 5
	lrQG = lrQG
	lrQD = lrQD
	cont_lamda = np.asarray([[1,1,1,1,1]],dtype=np.float32)
	lam = np.array_str(cont_lamda).replace(' ','_').replace('[','').replace(']','').replace('.','')
	dis = 0 #Ã»Ð´²»¶¯
	
	X_train = bonemarrow_cell()
	X_label = torch.LongTensor(np.zeros((X_train.shape[0]),dtype=int))
	X_train = torch.FloatTensor(X_train)
	train = torch.utils.data.TensorDataset(X_train,X_label)
	train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batchsize)
	dataiter = iter(train_loader)
	
	netG = _netG(nz = rand+dis*10+cont)
	print (netG)

	netD = _netD(nz = rand+dis*10+cont)
	print (netD)
	
	netD_D = _netD_D()
	netD_Q_3 = _netD_Q_3(nc = cont)

	netD, netG, netD_D, netD_Q_3 = netD.cuda(), netG.cuda(), netD_D.cuda(),  netD_Q_3.cuda()
	
	netG.apply(weights_init)
	netD.apply(weights_init)
	#netD_Q.apply(weights_init)
	netD_Q_3.apply(weights_init)
	netD_D.apply(weights_init)
	
	optimizerD = optim.RMSprop([
					{'params': netD.parameters()},
					{'params': netD_D.parameters()}
				], 0.00005)

	optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)
	 
	optimizerQ_G = optim.RMSprop([
					{'params': netG.parameters()},            
				], lrQG)

	optimizerQ_D = optim.RMSprop([
					{'params': netD.parameters()},
					{'params': netD_Q_3.parameters()}
				], lrQD)

	input = torch.FloatTensor(batchsize, 3, 32, 32)
	noise = torch.FloatTensor(batchsize, rand+10*dis+cont,1 ,1 )

	fixed_noise = torch.FloatTensor(np.random.multinomial(batchsize, 10*[0.1], size=1))
	c = torch.randn(batchsize, 10)
	c2 = torch.randn(batchsize, 10)
	c3 = torch.FloatTensor(np.random.uniform(-1,1,(batchsize,cont)))
	z = torch.randn(batchsize, rand)

	label = torch.FloatTensor(1)

	real_label = 1
	fake_label = 0

	criterion = nn.BCELoss()
	criterion_logli = nn.NLLLoss()
	criterion_mse = nn.MSELoss()

	criterion, criterion_logli, criterion_mse = criterion.cuda(), criterion_logli.cuda(), criterion_mse.cuda()
	input, label = input.cuda(), label.cuda()
	noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
	z, c, c2, c3 = z.cuda(), c.cuda(), c2.cuda(), c3.cuda()

	one = torch.FloatTensor([1])
	mone = one * -1
	one = one.cuda()
	mone = mone.cuda()
	cont_lamda = Variable(torch.from_numpy(cont_lamda).cuda(),requires_grad=False)
	
	gen_iterations = 0

	for epoch in range(num_epoch):

		dataiter = iter(train_loader)
		i = 0
		
		while i < len(train_loader):
			weight_clamp()
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
				weight_clamp()
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
				#rand_c,label_c = sample_c(batchsize)
				#c.copy_(rand_c)
				c3.uniform_(-1,1)
				noise = torch.cat([c3,z],1)
				noise_resize = noise.view(batchsize,rand+10*dis+cont,1,1)
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
			#Q_c_given_x = netD_Q(netD(G_sample)).view(batchsize, 10)
			Q_c_given_x_3 = netD_Q_3(netD(G_sample))
			
			#crossent_loss = criterion_logli(Q_c_given_x ,Variable(label_c.cuda()))
		   # print (Q_c_given_x)
			#crossent_loss_3 = criterion_mse(Q_c_given_x_3, Variable(c3)) 
			square_loss = (Q_c_given_x_3 - Variable(c3).view(1,-1,1,1))**2*cont_lamda
			square_loss = square_loss.sum()
			
			# ent_loss = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))
		   # ent_loss_2 = torch.mean(-torch.sum(c2 * torch.log(c2 + 1e-8), dim=1))
		   # ent_loss_3 = torch.mean(-torch.sum(c3 * torch.log(c3 + 1e-8), dim=1))

			#mi_loss = 0.1*crossent_loss  + 1*square_loss
			
			mi_loss = 1*square_loss
			mi_loss.backward()
			optimizerQ.step()
			
			if gen_iterations % 20 == 0 :
				errD = D_real - D_fake
				with open("/data/output_cell_{0}_{1}_{2}.txt".format(lrG, lrD, lam),'a+') as f:
					f.write('{0} {1} {2} {3}'.format(epoch, gen_iterations , -errD.data[0] , mi_loss.data[0]) + '\n')
				#print ('{0} {1} {2} {3}'.format(epoch, gen_iterations , -errD.data[0] , mi_loss.data[0]))
				
				#vutils.save_image(G_sample.data, '{0}fake_samples_{1}.png'.format(-errD.data[0], gen_iterations))
				vutils.save_image(G_sample.data, 'fake_samples.png',normalize = True)
				
				for t in range(0,cont):
					fixed_noise = generate_fix_noise_2(cont, rand)[t].reshape(100,rand+dis*10+cont,1,1)
					G_sample = netG(Variable(torch.FloatTensor(fixed_noise).cuda()))
					vutils.save_image(G_sample.data, '/data/map_{0}_{1}_{2}_{3}_{4}_cell.png'.format(t,epoch , lrG, lrD, lam),nrow=10,normalize=True)
	
	if epoch%10 == 0:
		torch.save(netG.state_dict(), '/data/netG_epoch_{0}_{1}_{2}_{3}.pth'.format(epoch, lrG, lrD, lam))
		torch.save(netD.state_dict(), '/data/netD_epoch_{0}_{1}_{2}_{3}.pth'.format(epoch, lrG, lrD, lam))
		torch.save(netD_D.state_dict(), '/data/netD_D_epoch_{0}_{1}_{2}_{3}.pth'.format(epoch, lrG, lrD, lam))
		torch.save(netD_Q_3.state_dict(), '/data/netD_Q_3_epoch_{0}_{1}_{2}_{3}.pth'.format(epoch, lrG, lrD, lam))
		
		