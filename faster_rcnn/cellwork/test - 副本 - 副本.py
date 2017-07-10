import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
#from faster_rcnn.faster_rcnn import nms_detections

import os
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

            runtime = t.toc()
            print('total spend: {}s'.format(runtime))
            lst.append(np.concatenate([(bbox[:,0]+n*225).reshape(-1,1),(bbox[:,1]+m*225).reshape(-1,1),(bbox[:,2]+n*225).reshape(-1,1),(bbox[:,3]+m*225).reshape(-1,1)],axis=1))

    im2show = np.copy(img[:,:,:])
    for i, element in enumerate(lst):
        for i, det in enumerate(element):
            det = tuple(int(x) for x in det)
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
    #cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    #1.0, (0, 0, 255), thickness=1)
	save=os.path.join('look', 'look.jpg')				
    cv2.imwrite(save,im2show)
'''
import matplotlib.pyplot as plt
from PIL import Image
plt.imshow(Image.open(os.path.join('demo', 'outaug_lr_pca_mut_70000_02.jpg')))
plt.show()
'''