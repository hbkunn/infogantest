from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel

import argparse
import random
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
import os
import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
import os
from itertools import combinations, permutations
import copy

#coherent_array must be 5*5
def cout_auc(coherent_array,cla_num):
    
    Accuracy =[]
    Position=[]
    
    if cla_num == 4:
        for i in range(0,4):
            for j in range(0,4):
                for m in range(0,4):
                    lis=[0,1,2,3]
                    if i!=j and i!=m and m!=j:
                        lis.remove(i)
                        lis.remove(j)
                        lis.remove(m)
                        a=lis[0]
                        accuracy = float(coherent_array[0][i]+coherent_array[1][j]+coherent_array[2][m]+coherent_array[3][a])/coherent_array.sum()
                        position = [i,j,m,a]
                        Accuracy.append(accuracy)
                        Position.append(position)
    ##case 5  
    if cla_num == 5:
        for i in range(0,5):
            for j in range(0,5):
                for m in range(0,5):
                    lis=[0,1,2,3,4]
                    if i!=j and i!=m and m!=j:
                        lis.remove(i)
                        lis.remove(j)
                        lis.remove(m)
                        a=lis[0]
                        b=lis[1]
                        accuracy = float(coherent_array[0][i]+coherent_array[1][j]+coherent_array[2][m]+coherent_array[3][a]+coherent_array[3][b])/coherent_array.sum()
                        position = [i,j,m,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)
        for i in range(0,5):
            for j in range(0,5):
                for m in range(0,5):
                    lis=[0,1,2,3,4]
                    if i!=j and i!=m and m!=j:
                        lis.remove(i)
                        lis.remove(j)
                        lis.remove(m)
                        a=lis[0]
                        b=lis[1]
                        accuracy = float(coherent_array[0][i]+coherent_array[1][j]+coherent_array[3][m]+coherent_array[2][a]+coherent_array[2][b])/coherent_array.sum()
                        position = [i,j,m,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)
        for i in range(0,5):
            for j in range(0,5):
                for m in range(0,5):
                    lis=[0,1,2,3,4]
                    if i!=j and i!=m and m!=j:
                        lis.remove(i)
                        lis.remove(j)
                        lis.remove(m)
                        a=lis[0]
                        b=lis[1]
                        accuracy = float(coherent_array[0][i]+coherent_array[3][j]+coherent_array[2][m]+coherent_array[1][a]+coherent_array[1][b])/coherent_array.sum()
                        position = [i,j,m,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)
        for i in range(0,5):
            for j in range(0,5):
                for m in range(0,5):
                    lis=[0,1,2,3,4]
                    if i!=j and i!=m and m!=j:
                        lis.remove(i)
                        lis.remove(j)
                        lis.remove(m)
                        a=lis[0]
                        b=lis[1]
                        accuracy = float(coherent_array[0][i]+coherent_array[3][j]+coherent_array[2][m]+coherent_array[0][a]+coherent_array[0][b])/coherent_array.sum()
                        position = [i,j,m,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)
                        
    ##case 6                    
    if cla_num == 6:    
        for i in range(0,6):
            for j in range(0,6):
                lis=[0,1,2,3,4,5]
                if i!=j:
                    lis.remove(i)
                    lis.remove(j)
                    lis1 = list(combinations(lis, 2))
                    for m,n in lis1:
                        lis2=copy.deepcopy(lis)
                        lis2.remove(m)
                        lis2.remove(n)
                        a = lis2[0]
                        b = lis2[1]
                        liss = [0,1,2,3]
                        liss1 = list(combinations(liss, 2))
                        for num in range(0,len(liss1)):
                            liss = [0,1,2,3]
                            h1 = liss1[num][0]
                            h2 = liss1[num][1]
                            liss.remove(h1)
                            liss.remove(h2)
                            h3 = liss[0]
                            h4 = liss[1]
                            accuracy = float(coherent_array[h1][i]+coherent_array[h2][j]+coherent_array[h3][m]+coherent_array[h3][n]+
                                             coherent_array[h4][a]+coherent_array[h4][b])/coherent_array.sum()
                            position = [i,j,m,n,a,b]
                            Accuracy.append(accuracy)
                            Position.append(position)   
                            
    ##case 7                    
    if cla_num == 7:    
        for T in range(0,7):
            lis=[0,1,2,3,4,5,6]
            lis.remove(T)
            lis1 = list(combinations(lis, 2))
            for i,j in lis1:
                lisij = copy.deepcopy(lis)
                lisij.remove(i)
                lisij.remove(j)
                lis2 = list(combinations(lisij, 2))
                for m,n in lis2:
                    lismn = copy.deepcopy(lisij)
                    lismn.remove(m)
                    lismn.remove(n)
                    a= lismn[0]
                    b= lismn[1]
                    liss = [0,1,2,3]
                    liss1 = list(combinations(liss, 3))
                    for h1,h2,h3 in liss1:
                        liss = [0,1,2,3]
                        liss.remove(h1)
                        liss.remove(h2)
                        liss.remove(h3)
                        h4 = liss[0]
                        accuracy = float(coherent_array[h1][T]+coherent_array[h2][i]+coherent_array[h2][j]+coherent_array[h3][m]+coherent_array[h3][n]+
                                         coherent_array[h4][a]+coherent_array[h4][b])/coherent_array.sum()
                        position = [i,j,m,n,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)                               

    ##case 8                    
    if cla_num == 8:    
        lis=[0,1,2,3,4,5,6,7]
        lis3 = list(combinations(lis, 2))
        for T1,T2 in lis3:
            lis=[0,1,2,3,4,5,6,7]
            lis.remove(T1)
            lis.remove(T2)
            lis1 = list(combinations(lis, 2))
            for i,j in lis1:
                lisij = copy.deepcopy(lis)
                lisij.remove(i)
                lisij.remove(j)
                lis2 = list(combinations(lisij, 2))
                for m,n in lis2:
                    lismn = copy.deepcopy(lisij)
                    lismn.remove(m)
                    lismn.remove(n)
                    a= lismn[0]
                    b= lismn[1]
                    liss = [0,1,2,3]
                    liss1 = list(combinations(liss, 3))
                    h1=0
                    h2=1
                    h3=2
                    h3=3
                    accuracy = float(coherent_array[h1][T1]+coherent_array[h1][T2]+coherent_array[h2][i]+coherent_array[h2][j]+coherent_array[h3][m]+coherent_array[h3][n]+
                                 coherent_array[h4][a]+coherent_array[h4][b])/coherent_array.sum()
                    position = [i,j,m,n,a,b]
                    Accuracy.append(accuracy)
                    Position.append(position) 
    ##case 9                    
    if cla_num == 9:    
        lis=[0,1,2,3,4,5,6,7,8]
        lis3 = list(combinations(lis, 3))
        for T1,T2,T3 in lis3:
            lis=[0,1,2,3,4,5,6,7,8]
            lis.remove(T1)
            lis.remove(T2)
            lis.remove(T3)
            lis1 = list(combinations(lis, 2))
            for i,j in lis1:
                lisij = copy.deepcopy(lis)
                lisij.remove(i)
                lisij.remove(j)
                lis2 = list(combinations(lisij, 2))
                for m,n in lis2:
                    lismn = copy.deepcopy(lisij)
                    lismn.remove(m)
                    lismn.remove(n)
                    a= lismn[0]
                    b= lismn[1]
                    liss = [0,1,2,3]
                    liss1 = list(combinations(liss, 1))
                    liss = [0,1,2,3]
                    for h1 in range(0,4):
                        liss.remove(h1)
                        h2=liss[0]
                        h3=liss[1]
                        h4=liss[2]
                        accuracy = float(coherent_array[h1][T1]+coherent_array[h1][T2]+coherent_array[h2][i]+coherent_array[h2][j]+coherent_array[h3][m]+coherent_array[h3][n]+coherent_array[h4][a]+coherent_array[h4][b])/coherent_array.sum()
                        position = [i,j,m,n,a,b]
                        Accuracy.append(accuracy)
                        Position.append(position)    
    auc = max(Accuracy)
    p=Accuracy.index(auc)
    pos = Position[p]
    print(auc)
    
    
def clustering(paramsG,paramsD,paramsDD,paramsDQ,num_dis_category):
    root_dir = '/disk1/labeled/'
    npyList = os.listdir(root_dir)
    npyList = [root_dir+n for n in npyList]
    result = []
    label = []
    for n,array in enumerate(npyList):
        result.append(np.load(array))
        label.append([n]*result[n].shape[0])

    result = np.concatenate(result)
    label = np.concatenate(label)

    X = np.asarray([x.transpose((2,0,1)) for x in result])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    X_train = torch.FloatTensor(X)
    X_label = torch.LongTensor(label)
    train = torch.utils.data.TensorDataset(X_train,X_label)
    train_loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=1)
    
    rand=128
    dis=1
    dis_category = num_dis_category

    class avgpool(nn.Module):
        def __init__(self, up_size=0):
            super(avgpool, self).__init__()

        def forward(self, x):
            out_man = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4
            return out_man

    class ResidualBlock(nn.Module):

        def __init__(self, in_dim, out_dim, resample=None, up_size=0):
            super(ResidualBlock, self).__init__()

            if resample == 'up':
                self.bn1 = nn.BatchNorm2d(in_dim)
                self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
                self.upsample = torch.nn.Upsample(up_size,2)
                self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
                self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
                self.bn2 = nn.BatchNorm2d(out_dim)

            elif resample == 'down':
                self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
                self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
                self.pool = avgpool()
                self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)

            elif resample == None:
                self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
                self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)

            self.resample = resample

        def forward(self, x):

            if self.resample == None:
                shortcut = x
                output = x

                output = nn.functional.relu(output)
                output = self.conv1(output)
                output = nn.functional.relu(output)
                output = self.conv2(output)

            elif self.resample == 'up':
                shortcut = x
                output = x

                shortcut = self.upsample(shortcut) #upsampleconv
                shortcut = self.upsample_conv(shortcut)

                output = self.bn1(output)
                output = nn.functional.relu(output)
                output = self.conv1(output)

                output = self.bn2(output)
                output = nn.functional.relu(output)
                output = self.upsample(output) #upsampleconv
                output = self.conv2(output)

            elif self.resample == 'down':
                shortcut = x
                output = x

                shortcut = self.pool_conv(shortcut) #convmeanpool
                shortcut = self.pool(shortcut)

                output = nn.functional.relu(output)
                output = self.conv1(output)

                output = nn.functional.relu(output)
                output = self.conv2(output)    #convmeanpool
                output = self.pool(output)

            return output+shortcut

    class ResidualBlock_thefirstone(nn.Module):

        def __init__(self, in_dim, out_dim, resample=None, up_size=0):
            super(ResidualBlock_thefirstone, self).__init__()

            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.pool = avgpool()
            self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)

        def forward(self, x):

            shortcut = x
            output = x

            shortcut = self.pool(shortcut) #meanpoolconv
            shortcut = self.pool_conv(shortcut)

            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output) #convmeanpool
            output = self.pool(output)

            return output+shortcut


    class generator(nn.Module):

        def __init__(self, rand=128):
            super(generator, self).__init__()
            self.rand = rand
            self.linear = nn.Linear(rand  ,2048, bias=True)
            self.layer_up_1 = ResidualBlock(128, 128, 'up', up_size=8)
            self.layer_up_2 = ResidualBlock(128, 128, 'up', up_size=16)
            self.layer_up_3 = ResidualBlock(128, 128, 'up', up_size=32)
            self.bn1 = nn.BatchNorm2d(128)
            self.conv_last = nn.Conv2d(128, 3, 3, 1, 1, bias=True)

        def forward(self, x):
            x = x.view(-1,self.rand)
            x = self.linear(x)
            x = x.view(-1,128,4,4)
            x = self.layer_up_1(x)
            x = self.layer_up_2(x)
            x = self.layer_up_3(x)
            x = self.bn1(x)
            x = nn.functional.relu(x)
            x = self.conv_last(x)
            x = nn.functional.tanh(x)
            return x

    netG = generator(rand = rand+dis*dis_category)

    class discriminator(nn.Module):

        def __init__(self):
            super(discriminator, self).__init__()
            self.layer_down_1 = ResidualBlock_thefirstone(3, 128)
            self.layer_down_2 = ResidualBlock(128, 128, 'down')
            self.layer_none_1 = ResidualBlock(128, 128, None)
            self.layer_none_2 = ResidualBlock(128, 128, None)
            #self.mean_pool = nn.AvgPool2d(8,1,0)
            #self.linear = nn.Linear(128,1, bias=True)
            #self.linear2 = nn.Linear(128,10, bias=True)

        def forward(self, x):
            x = self.layer_down_1(x)
            x = self.layer_down_2(x)
            x = self.layer_none_1(x)
            x = self.layer_none_2(x)
            #x = self.mean_pool(x)
            x = nn.functional.relu(x)
            x = x.mean(2).mean(2)
            x = x.view(-1, 128)

            #shortcut = x
            #output = x

            #output = self.linear(output)
            #shortcut= self.linear2(shortcut)

            #return output.view(-1,1,1,1), shortcut.view(-1,10,1,1)
            return x

    netD = discriminator()

#torch.cuda.set_device(1)

    class _netD_D(nn.Module):
        def __init__(self):
            super(_netD_D, self).__init__()
            self.linear = nn.Linear(128,1, bias=True)
            #self.conv = nn.Conv2d(4096, 1, 1, 1, 0, bias=True)

        def forward(self, x):
            x = self.linear(x)
            return x.view(-1,1,1,1)

    class _netD_Q(nn.Module):
        def __init__(self, nd = 10):
            super(_netD_Q, self).__init__()
            # input is Z, going into a convolution
            #self.conv = nn.Conv2d(4096, 128, 1, 1, 0, bias=True)
            #self.relu = nn.LeakyReLU(0.2, inplace=True)
            #self.conv2 = nn.Conv2d(128, nd, 1, 1, 0, bias=True)
            self.softmax = nn.LogSoftmax()
            #self.linear1 = nn.Linear(4096,128, bias=True)
            #self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.linear2 = nn.Linear(128,nd, bias=True)
            self.nd = nd

        def forward(self, x):
            #x = self.linear1(x)
            #x = self.relu(x)
            x = self.linear2(x)
            x = self.softmax(x)
           # x = x.view(64,10)
            return x.view(-1,self.nd,1,1)

    netD_D = _netD_D()
    netD_Q = _netD_Q(dis_category)
    
    
    netG.load_state_dict(torch.load(paramsG))
    netD.load_state_dict(torch.load(paramsD))
    netD_D.load_state_dict(torch.load(paramsDD))
    netD_Q.load_state_dict(torch.load(paramsDQ))
    
    data_iter = iter(train_loader)
    predict = []
    netD = netD.cuda()
    netD_Q = netD_Q.cuda()

    for iteration in data_iter:
        img, img_label = iteration
        predict_label = netD_Q(netD(Variable(img.cuda())))
        predict.append(predict_label.data.cpu().numpy())    
        
    predict_label = []

    for n in range(0, len(predict)):
        predict_label.append(np.argmax(predict[n]))

    coherent_array = np.zeros((5,5),dtype=int)

    for n in range(0, len(predict)):
        coherent_array[label[n],predict_label[n]] +=1
        
    cout_auc(coherent_array,num_dis_category)
    
    

