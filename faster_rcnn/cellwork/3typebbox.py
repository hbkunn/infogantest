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
model_file = '/home/tangye/faster_rcnn_pytorch/data/faster_rcnn_mir_ro90_4_1scale40000.h5'
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





def space(x1,x2,y1,y2):
    a=(y1-x1)*(y2-x2)
    return a
   
   
 ###################################
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
        if len(lst) == 0:
            continue
        #data_dict['im_data'] = (img-mean)[hight*200:hight*200+200,width*200:width*200+200,::-1].reshape(1,200,200,3).astype(np.float32)
        #store = np.asarray(lst,dtype=int).reshape(-1,2)
        q = 0
        #Bbox = np.zeros((len(lst),4),dtype=np.float32)
        for i in lst:
            q+=1
            x1,y1 = max(0,i[0]-16),max(0,i[1]-16)
            x2,y2 = x1+32,y1+32
            if x1+32>1200:
                x1, x2 = (1200-32), 1200
            if y1+32>1200:
                y1, y2 = (1200-32), 1200
            #biglst.append(image_output)
            bbox.append([y1,x1,y2,x2])               
im2show = np.copy(img[:,:,:])
 
 
 ###################################
Bbox=bbox
rightlst=[] 
for i, det in enumerate(pred_boxes):
    dets = tuple(int(x) for x in det)
    for n in range(len(Bbox)):
	   ##1 right
	   Bbox[n][0]=int(Bbox[n][0])
	   Bbox[n][1]=int(Bbox[n][1])
	   Bbox[n][2]=int(Bbox[n][2])
	   Bbox[n][3]=int(Bbox[n][3])
	   
	   s=space(Bbox[n][0],Bbox[n][1],Bbox[n][2],Bbox[n][3])
	   
       if det[0]<=Bbox[n][2] and det[0]>=Bbox[n][0] and det[1]<=Bbox[n][1] and det[1]>=Bbox[n][3] and space(det[0],det[1],Bbox[n][2],Bbox[n][3])<=s*0.5:
	      Bbox[n][0]=Bbox[n][1]=Bbox[n][2]=Bbox[n][3]=0
		  rightlst.append(dets)
          cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 205, 51), 2) 
       elif det[2]<=Bbox[n][2] and det[2]>=Bbox[n][0] and det[3]<=Bbox[n][1] and det[3]>=Bbox[n][3] and space(det[2],det[3],Bbox[n][0],Bbox[n][1])<=s*0.5:
	      Bbox[n][0]=Bbox[n][1]=Bbox[n][2]=Bbox[n][3]=0
          cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 205, 51), 2)
		  rightlst.append(dets)
       elif det[2]<=Bbox[n][2] and det[2]>=Bbox[n][0] and det[1]<=Bbox[n][1] and det[1]>=Bbox[n][3] and space(det[2],det[1],Bbox[n][0],Bbox[n][3])<=s*0.5:
          Bbox[n][0]=Bbox[n][1]=Bbox[n][2]=Bbox[n][3]=0
		  rightlst.append(dets)
		  cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 205, 51), 2)
       elif det[0]<=Bbox[n][2] and det[0]>=Bbox[n][0] and det[3]<=Bbox[n][1] and det[3]>=Bbox[n][3] and space(det[0],det[3],Bbox[n][2],Bbox[n][1])<=s*0.5:
          Bbox[n][0]=Bbox[n][1]=Bbox[n][2]=Bbox[n][3]=0
		  rightlst.append(dets)
		  cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 205, 51), 2)
	   ##2 wrong 
	   else 
	      cv2.rectangle(im2show, dets[0:2], dets[2:4], (255, 0, 0), 2)
	   ##miss
for n in range(len(Bbox)):
  if int(Bbox[n][0])!=0 or int(Bbox[n][1])!=0 or int(Bbox[n][1])!=0 or int(Bbox[n][1])!=0:   
  cv2.rectangle(im2show, (int(Bbox[n][0]),int(Bbox[n][1])),(int(Bbox[n][2]),int(Bbox[n][3])), (0, 255, 0), 2)

cv2.imwrite(os.path.join('demo', 'result.jpg'), im2show)
import matplotlib.pyplot as plt
from PIL import Image
plt.imshow(Image.open(os.path.join('demo', 'result.jpg')))
plt.show()  