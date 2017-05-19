    root_data='/home/tangye/faster_rcnn_pytorch/data/BM_GRAZ/source/BM_GRAZ_HE_0001_01.png'     
    root_label='/home/tangye/faster_rcnn_pytorch/data/BM_GRAZ/annotations/BM_GRAZ_HE_0001_01.png' 
    img = cv2.imread(root_data)    
    label = cv2.imread(root_label, 2)
    img = cv2.resize(img,(4800,4800))
    #label = cv2.resize(label,(4800,4800))
    for hight in range(0,6):
        for width in range(0,6):
            data_dict = {}
            lst = []
            for x in range(max(0,hight*200-16),min(1200,hight*200+200+16)):
                for y in range(max(0,width*200-16),min(1200,width*200+200+16)):
                    if label[x,y] == 255:
                        lst.append([(x-hight*200)*4,(y-width*200)*4])
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
                if x1+32>800:
                    x1, x2 = (800-32), 800
                if y1+32>800:
                    y1, y2 = (800-32), 800
                #biglst.append(image_output)
                bbox.append([y1,x1,y2,x2])
            im2show=np.copy(image[800*hight:800*hight+800,800*width:800*width+800,::-1]
            for n in range(len(bbox)):
                cv2.rectangle(im2show, (int(bbox[n][0]),int(bbox[n][1])),(int(bbox[n][2]),int(bbox[n][3])), (255, 205, 51), 2)
    #cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                   #1.0, (0, 0, 255), thickness=1)
		    p=hight+width
            save=os.path.join('look', 'xiao4bei_{}'.format(p))
            cv2.imwrite(save, im2show)			