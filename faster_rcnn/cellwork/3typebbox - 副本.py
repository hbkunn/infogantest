import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.faster_rcnn import nms_detections

import os
im_file ='/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/BM_GRAZ_HE_0008_01.png'
    #im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
image = cv2.imread(im_file)
    #image = cv2.resize(image,(4800,4800))
#img = image-mean
    #print (img)
model_file = '/home/tangye/faster_rcnn_pytorch/data/faster_rcnn_aug10000hao.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
detector = FasterRCNN(classes=np.asarray(['__background__','cell']))
network.load_net(model_file, detector)
detector.cuda()
detector.eval()
print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')
    
t = Timer()
t.tic()
    
lst = []
count=0
for m in range(0,5):
    for n in range(0,5):
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
        dets, scores, classes = detector.detect(image[m*225:300+m*225,n*225:300+n*225,::-1], 0.1)
        if dets.shape[0] == 0:
            continue
        runtime = t.toc()
        print('total spend: {}s'.format(runtime))
        lst.append(np.concatenate([(dets[:,0]+n*225).reshape(-1,1),(dets[:,1]+m*225).reshape(-1,1),(dets[:,2]+n*225).reshape(-1,1),
                                   (dets[:,3]+m*225).reshape(-1,1),scores.reshape(-1,1)],axis=1))
        count += dets.shape[0]

keep_boxes = np.zeros((count,4),dtype=np.float32)
keep_scores = np.zeros((count),dtype=np.float32)
i = 0
for element in lst:
    for det in element:
        keep_boxes[i] = det[0:4]
        keep_scores[i] = det[4]
        i+=1

pred_boxes, scores = nms_detections(keep_boxes, keep_scores, nms_thresh = 0.2, inds=None)
print '##step one finish##'
    #cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    #1.0, (0, 0, 255), thickness=1)


#########################################
bbox=[]
xylst=[]
img = cv2.imread('/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/BM_GRAZ_HE_0008_01.png')    
label = cv2.imread('/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/annotations/BM_GRAZ_HE_0008_01.png', 2)
for hight in range(0,6):
    for width in range(0,6):
        data_dict = {}
        lst = []
        for x in range(max(0,hight*200-16),min(1200,hight*200+200+16)):
            for y in range(max(0,width*200-16),min(1200,width*200+200+16)):
                if label[x,y] == 255:
                    lst.append([x,y])
                    xylst.append([x,y])
                    xy=xylst
im2show = np.copy(img[:,:,:])
print '##step two finish##'


#######################
img = cv2.imread('/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/BM_GRAZ_HE_0008_01.png')  
im2show = np.copy(img[:,:,:])
rightlst=[]
for i, det in enumerate(pred_boxes):
    dets = tuple(int(x) for x in det)
    for n in range(len(xylst)):
       ##1 right
        xylst[n][0]=int(xylst[n][0])
        xylst[n][1]=int(xylst[n][1])
        #s=space(bbox[n][0],bbox[n][1],bbox[n][2],bbox[n][3])

        if dets[0]<=xylst[n][1] and dets[2]>=xylst[n][1] and dets[1]<=xylst[n][0] and dets[3]>=xylst[n][0]:
           # bbox[n][0]=bbox[n][1]=bbox[n][2]=bbox[n][3]=0
            #print n
            #print 'hhh'
           # print dets
           # print xylst[n][1]
           # print xylst[n][0]
            xylst[n]=[0,0]
            rightlst.append(dets)
            #print rightlst
            cv2.rectangle(im2show, dets[0:2], dets[2:4], (0, 0, 0), 2) 
        #else:
           # cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 0, 0), 2)
print 'length of rightlst'            
print len(rightlst)
####### wrong
for i, det in enumerate(pred_boxes):
    dets = tuple(int(x) for x in det)
    for y in range(len(rightlst)):
        #b=(pred_boxes[xx]==rightlst[y])
        if dets==rightlst[y]:
            pred_boxes[i]=[0,0,0,0]
for i, ddet in enumerate(pred_boxes):
    ddets = tuple(int(x) for x in ddet)
    cv2.rectangle(im2show, ddets[0:2], ddets[2:4], (0, 255, 0), 2)

print '##step three finish##'

###################
#data_dict['im_data'] = (img-mean)[hight*200:hight*200+200,width*200:width*200+200,::-1].reshape(1,200,200,3).astype(np.float32)
#store = np.asarray(lst,dtype=int).reshape(-1,2)
q = 0
bbox=[]
#Bbox = np.zeros((len(lst),4),dtype=np.float32)
for i in xylst:
    if i!=[0,0]:
        q+=1
        x1,y1 = max(0,i[0]-16),max(0,i[1]-16)
        x2,y2 = x1+32,y1+32
        if x1+32>1200:
            x1, x2 = (1200-32), 1200
        if y1+32>1200:
            y1, y2 = (1200-32), 1200
        #biglst.append(image_output)
        bbox.append([y1,x1,y2,x2])
print 'length of bbox:'
print (len(bbox))
for n in range(len(bbox)):
    cv2.rectangle(im2show, (int(bbox[n][0]),int(bbox[n][1])),(int(bbox[n][2]),int(bbox[n][3])), (0, 0, 255), 2)
print 'step four finish'

cv2.imwrite(os.path.join('demo', 'resultRI.jpg'), im2show)
import matplotlib.pyplot as plt
from PIL import Image
plt.imshow(Image.open(os.path.join('demo', 'resultRI.jpg')))
plt.show()