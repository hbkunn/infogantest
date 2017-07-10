import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
#from faster_rcnn.faster_rcnn import nms_detections

import os
im_file ='/home/hbkunn/rpn/faster_rcnn_pytorch/data/BM_GRAZ/source/BM_GRAZ_HE_0008_01.png'
    #im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
image = cv2.imread(im_file)
    #image = cv2.resize(image,(4800,4800))
#img = image-mean
    #print (img)
model_file = '/home/tangye/faster_rcnn_pytorch/data/faster_rcnn_12370000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
detector = FasterRCNN(classes=np.asarray(['__background__','cell']))
network.load_net(model_file, detector)
detector.cuda()
detector.eval()
print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')s
    
t = Timer()
t.tic()
    
lst = []
for m in range(0,5):
    for n in range(0,5):
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
        dets, scores, classes = detector.detect(image[m*225:300+m*225,n*225:300+n*225,::-1], 0.1)
        if dets.shape[0] == 0:
            continue
        runtime = t.toc()
        print('total spend: {}s'.format(runtime))
        lst.append(np.concatenate([(dets[:,0]+n*225).reshape(-1,1),(dets[:,1]+m*225).reshape(-1,1),(dets[:,2]+n*225).reshape(-1,1),(dets[:,3]+m*225).reshape(-1,1)],axis=1))

im2show = np.copy(image[:,:,:])
for i, element in enumerate(lst):
    for i, det in enumerate(element):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
    #cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    #1.0, (0, 0, 255), thickness=1)
cv2.imwrite(os.path.join('demo', 'outaug_lr_pca_mut_70000_02.jpg'), im2show)
import matplotlib.pyplot as plt
from PIL import Image
plt.imshow(Image.open(os.path.join('demo', 'outaug_lr_pca_mut_70000_02.jpg')))
plt.show()