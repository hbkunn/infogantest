
# coding: utf-8

# In[1]:

import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# In[2]:

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'#??有用么？？
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
output_dir = 'models/saved_model3'

start_step = 0
end_step = 100000
lr_decay_steps = {80000, 100000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=np.asarray(['__background__','cell']), debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
# network.load_net(model_file, net)
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)
#都是在准备吧
 
net.cuda()
net.train()#告诉它要开始训练了

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
#貌似还是在准备,这之前是在做什么

# In[3]:

output_dir = 'data'
    
use_tensorboard = None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()


# In[2]:

import os
import cv2
import numpy as np

root_data = '/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/'
root_label = '/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/annotations/'
imageList = os.listdir(root_data)

#计算图像均值
mean = np.zeros((3),dtype=np.float32)
for i in imageList[0:-1]:
    img = cv2.imread(root_data + i)
    mean = (mean + img.mean(axis=0).mean(axis=0)/len(imageList[0:-1]))
    
print (mean)


# In[34]:

import numpy as np
import matplotlib.pyplot as plt

root_data = '/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/'
root_label = '/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/annotations/'
imageList = os.listdir(root_data)

iterator = []
biglst = []
for i in imageList[0:-1]:
    img = cv2.imread(root_data + i)    
    label = cv2.imread(root_label + i, 2)
    for hight in range(0,6):#图片是1200*1200，按照200*200的滑动窗去找标签
        for width in range(0,6):
            data_dict = {}
            lst = []
            for x in range(hight*200,hight*200+200):
                for y in range(width*200,width*200+200):
                    if label[x,y] == 255:
                        lst.append([x-hight*200,y-width*200])#截成200*200的小图去训练，每个图重新建系，所以要减

            if len(lst) == 0:
                continue
            data_dict['im_data'] = (img-mean)[hight*200:hight*200+200,width*200:width*200+200,::-1].reshape(1,200,200,3).astype(np.float32)#，格式装换，字典中im_data的映射为一个200*200的图像
            #store = np.asarray(lst,dtype=int).reshape(-1,2)
            q = 0
            bbox = np.zeros((len(lst),5),dtype=np.float32)
            for i in lst:#lst是标签的坐标，即所截成的200*200小图中的亮点位置坐标
                q+=1
                x1,y1 = max(0,i[0]-16),max(0,i[1]-16)
                x2,y2 = x1+32,y1+32
                if x1+32>200:
                    x1, x2 = (200-32), 200
                if y1+32>200:
                    y1, y2 = (200-32), 200
                #biglst.append(image_output)
                bbox[q-1] =  [y1,x1,y2,x2,1] #获取每个点的小框32*32，每个亮点都在中心
            assert q == len(lst)
            data_dict['gt_boxes'] = bbox.astype(np.float32)
            iterator.append(data_dict)#这样就可以把所有数据的标签存在一个iterator里面了
            '''
            im2show = np.copy(img[hight*200:hight*200+200,width*200:width*200+200,:])
            for i, det in enumerate(bbox.astype(np.float32)):
                det = tuple(int(x) for x in det)
                cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.imwrite(os.path.join('store_for_effect', str(len(iterator))+'.jpg'), im2show)
            '''
            
for i in imageList[0:-1]:
    print (len(iterator))
    img = cv2.imread(root_data + i)    
    label = cv2.imread(root_label + i, 2)
    img = cv2.resize(img,(4800,4800))
    #label = cv2.resize(label,(4800,4800))
    for hight in range(0,6):
        for width in range(0,6):
            data_dict = {}
            lst = []
            for x in range(hight*200,hight*200+200):
                for y in range(width*200,width*200+200):
                    if label[x,y] == 255:
                        lst.append([(x-hight*200)*4,(y-width*200)*4])#被resize大了4倍

            if len(lst) == 0:
                continue
            data_dict['im_data'] = (img-mean)[hight*800:hight*800+800,width*800:width*800+800,::-1].reshape(1,800,800,3).astype(np.float32)#800*800图片
            #store = np.asarray(lst,dtype=int).reshape(-1,2)
            q = 0
            bbox = np.zeros((len(lst),5),dtype=np.float32)
            for i in lst:
                q+=1
                x1,y1 = max(0,i[0]-64),max(0,i[1]-64)
                x2,y2 = x1+128,y1+128
                if x1+128>800:
                    x1, x2 = (800-128), 800
                if y1+128>800:
                    y1, y2 = (800-128), 800
                #biglst.append(image_output)
                bbox[q-1] =  [y1,x1,y2,x2,1] 
            assert q == len(lst)
            data_dict['gt_boxes'] = bbox.astype(np.float32)
            iterator.append(data_dict)
            '''
            im2show = np.copy(img[hight*200:hight*200+200,width*200:width*200+200,:])
            for i, det in enumerate(bbox.astype(np.float32)):
                det = tuple(int(x) for x in det)
                cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.imwrite(os.path.join('store_for_effect', str(len(iterator))+'.jpg'), im2show)
            '''
            
print (len(iterator))#iterator=【【im_data，gt_boxes】，【】，【】，【】】get_box 是一个建议框矩阵而im_data可以看成是一个训练图有200*200,800*800两种，上面有很多建议框
# In[4]:

import itertools
import random

im_info = np.asarray([[200.,200.,1]],dtype=np.float32)
gt_ishard = None
dontcare_areas = np.zeros((0,4), dtype=np.float32)
#调用各种库开始训练，感觉前面就是做了一下标签
for step,element in enumerate(itertools.cycle(iterator)):
    if step%len(iterator) == 0 :
        random.shuffle(iterator)
    # get one batch
    #blobs = data_layer.forward()
    im_data = element.get('im_data')
    #im_info = blobs['im_info']
    gt_boxes = element.get('gt_boxes')
    #gt_ishard = blobs['gt_ishard']
    #dontcare_areas = blobs['dontcare_areas']
    size = im_data.shape[2]#111图像大小？
    im_info = np.asarray([[size,size,1]],dtype=np.float32)

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:#???
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % 1000 == 0:
        duration = t.toc(average=False)#111???t是啥
        fps = step_cnt / duration

        log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:#???
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True


    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False


# In[ ]:



