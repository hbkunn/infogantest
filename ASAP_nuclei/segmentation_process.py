import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color
import time

#Some nice default configuration for plots

def segmentation(image_name):
    
    inputImageFile = image_name
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    print("reading image done...")
    
    refImageFile = ('/disk1/rpn/data/BM_GRAZ/source/BM_GRAZ_HE_0001_01.png')
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    print("normalization done...")
    
    #unsupervised color deconvolution
    
    stainColorMap = {
        'hematoxylin': [0.64, 0.72, 0.27],
        'eosin':       [0.09, 0.95, 0.28],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }

    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains


    W_init = np.array([stainColorMap[stain_1],
                       stainColorMap[stain_2]]).T

    sparsity_factor = 0.5

    imDeconvolved, W_est = htk.preprocessing.color_deconvolution.sparse_color_deconvolution(
        imInput, W_init, sparsity_factor)
    
    print 'Estimated stain colors (in rows): '
    print W_est.T
    print("color deconvolution done...")

    imNucleiStain = 255 - imDeconvolved[:, :, 0]
    
    
    '''
    #supervised color deconvolution
    stainColorMap = {
        'hematoxylin': [0.59532715,0.71332775,0.36979604],
        'eosin':       [0.12681853,0.89384825,0.4300609],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T
    imDeconvolved = htk.preprocessing.color_deconvolution.color_convolution(imNmzd, W)
    imNucleiStain = imDeconvolved[:, :, 1]
    print("color deconvolution done...")
    '''
    
    foreground_threshold = 210
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
    imNucleiStain < foreground_threshold)
    
    min_radius = 3
    max_radius = 15

    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 5

    imNucleiSegMask, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 80

    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(imNucleiSegMask)

    print 'Number of nuclei = ', len(objProps)
    
    size = 32
    output = np.ones((size,size,3),dtype=np.uint8)*255
    output_convex = np.zeros((size,size,1),dtype=bool)
    image_dict = {}
   
    #segmentate
    '''
    for i,n in enumerate(objProps):
        w, h = int(n.centroid[0]),int(n.centroid[1])
        w_convex, h_convex = int(n.convex_image.shape[0]), int(n.convex_image.shape[1])
        if min(w-size/2,h-size/2)<=0 or max(w+size/2,h+size/2)>=1200:
            continue
        if min(size/2-w_convex/2,  size/2-h_convex/2)<=0 or max(size/2-w_convex/2+w_convex, size/2-h_convex/2+h_convex) >=size:
            continue
        output[: ,: ,:] = imInput[w-size/2:w+size/2,h-size/2:h+size/2,:]
        output_convex[size/2-w_convex/2:size/2-w_convex/2+w_convex, size/2-h_convex/2:size/2-h_convex/2+h_convex, 0]  \
                = n.convex_image#.transpose(1,0)
        result = np.tile(output_convex, (1, 1, 3))*output
        image_dict[i] = result    

        output = np.ones((size,size,3),dtype=np.uint8)*255
        output_convex = np.zeros((size,size,1),dtype=bool)
    '''
    
    #resize
    for i,n in enumerate(objProps):
        w, h = int(n.centroid[0]),int(n.centroid[1])
        w_convex, h_convex = int(n.convex_image.shape[0]), int(n.convex_image.shape[1])
        if min(w-w_convex/2,h-w_convex/2)<=0 or max(w+h_convex/2,h+h_convex/2)>=1200:
            continue
        if min(output.shape)==0:
            continue
        output = imInput[w-w_convex/2:w+w_convex/2,h-h_convex/2:h+h_convex/2,:]
        output = sp.misc.imresize(output, (32,32))
        image_dict[i] = output
    image = np.array(image_dict.values())
    np.save('/disk1/cell_resize/{0}.npy'.format(inputImageFile[-9:-4]),image)
    print ('done')
    
    
def segmentation_0713(image_name):
    inputImageFile = image_name  # Easy1.png
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    refImageFile = ('/disk1/rpn/data/BM_GRAZ/source/BM_GRAZ_HE_0007_01.png')  # L1.png

    imReference = skimage.io.imread(refImageFile)[:, :, :3]

    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)

    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    print('load done')
    img = np.copy(imNmzd[:,:,:]).astype(np.uint8)
    #W = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(img, None, beta=0.2)
    #imDeconvolved = htk.preprocessing.color_deconvolution.color_convolution(imNmzd, W, I_0=0.0)
    #print('color1 done')

    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.64, 0.72, 0.27],
        'eosin':       [0.09, 0.95, 0.28],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains


    # create initial stain matrix
    W_init = np.array([stainColorMap[stain_1],
                       stainColorMap[stain_2]]).T

    # perform sparse color deconvolution
    sparsity_factor = 0.5

    imDeconvolved1, W_est = htk.preprocessing.color_deconvolution.sparse_color_deconvolution(
        imNmzd, W_init, sparsity_factor)
    imDeconvolved1 = imDeconvolved

    print 'Estimated stain colors (in rows): '
    print W_est.T

    imNucleiStain =imDeconvolved[:, :, 0]
    foreground_threshold =225

    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
    imNucleiStain < foreground_threshold)
    
    # run adaptive multi-scale LoG filter
    min_radius = 5
    max_radius = 35

    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 5

    imNucleiSegMask, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 100

    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(imNucleiSegMask)

    print 'Number of nuclei = ', len(objProps)
    
    size = 32
    output = np.ones((size,size,3),dtype=np.uint8)*0
    output_convex = np.zeros((size,size,1),dtype=bool)
    image_dict = {}
    plt.rcParams['figure.figsize'] = 1, 1

    count = 0
    deconvolution = np.expand_dims(imDeconvolved1[:, :, 0],axis=2)
    deconvolution = np.tile(deconvolution,(1,1,3))
    for i,n in enumerate(objProps):

        x, y = n.bbox[0],n.bbox[1]
        w, h = int(n.convex_image.shape[0]), int(n.convex_image.shape[1])
        x_center, y_center = int(n.centroid[0]),int(n.centroid[1])

        image = deconvolution[x: x+w,y:y+h,:]
        #image = imInput[x: x+w,y:y+h,:]
        convex_image = n.convex_image[0: w,0:h]
        convex_image = np.expand_dims(convex_image,axis=2)
        segmentate_image = np.tile(convex_image, (1, 1, 3))*image
        #segmentate_image = image

        if max(w,h)>32:
            scale = 32/float(max(w,h))
            w, h = int(w*scale), int(h*scale)
            segmentate_image = sp.misc.imresize(segmentate_image, (w, h))

        npad = ((16-w/2,32-w-(16-w/2)),(16-h/2,32-h-(16-h/2)),(0,0))
        segmentate_image = np.pad(segmentate_image, pad_width=npad,constant_values=0,mode='constant')
        image_dict[i] = segmentate_image
    
    image = np.array(image_dict.values())
    np.save('/disk1/cell_segmentate_decon/{0}.npy'.format(inputImageFile[-9:-4]),image)
    print ('done')
    
def segmentation_un_sparse(image_name):
    
    inputImageFile = image_name
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    print("reading image done...")
    
    refImageFile = ('/disk1/rpn/data/BM_GRAZ/source/BM_GRAZ_HE_0001_01.png')
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    print("normalization done...")
    
    #unsupervised color deconvolution
    
    stainColorMap = {
        'hematoxylin': [0.64, 0.72, 0.27],
        'eosin':       [0.09, 0.95, 0.28],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }

    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains


    W_init = np.array([stainColorMap[stain_1],
                       stainColorMap[stain_2]]).T

    sparsity_factor = 0.5

    imDeconvolved, W_est = htk.preprocessing.color_deconvolution.sparse_color_deconvolution(
        imInput, W_init, sparsity_factor)
    
    print 'Estimated stain colors (in rows): '
    print W_est.T
    print("color deconvolution done...")

    imNucleiStain = 255 - imDeconvolved[:, :, 0]
    
    
    '''
    #supervised color deconvolution
    stainColorMap = {
        'hematoxylin': [0.59532715,0.71332775,0.36979604],
        'eosin':       [0.12681853,0.89384825,0.4300609],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T
    imDeconvolved = htk.preprocessing.color_deconvolution.color_convolution(imNmzd, W)
    imNucleiStain = imDeconvolved[:, :, 1]
    print("color deconvolution done...")
    '''
    
    foreground_threshold = 220
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
    imNucleiStain < foreground_threshold)
    
    min_radius = 5
    max_radius = 35

    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10

    imNucleiSegMask, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 80

    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(imNucleiSegMask)

    print 'Number of nuclei = ', len(objProps)
    
    size = 32
    output = np.ones((size,size,3),dtype=np.uint8)*255
    output_convex = np.zeros((size,size,1),dtype=bool)
    image_dict = {}
   
    #segmentate
    '''
    for i,n in enumerate(objProps):
        w, h = int(n.centroid[0]),int(n.centroid[1])
        w_convex, h_convex = int(n.convex_image.shape[0]), int(n.convex_image.shape[1])
        if min(w-size/2,h-size/2)<=0 or max(w+size/2,h+size/2)>=1200:
            continue
        if min(size/2-w_convex/2,  size/2-h_convex/2)<=0 or max(size/2-w_convex/2+w_convex, size/2-h_convex/2+h_convex) >=size:
            continue
        output[: ,: ,:] = imInput[w-size/2:w+size/2,h-size/2:h+size/2,:]
        output_convex[size/2-w_convex/2:size/2-w_convex/2+w_convex, size/2-h_convex/2:size/2-h_convex/2+h_convex, 0]  \
                = n.convex_image#.transpose(1,0)
        result = np.tile(output_convex, (1, 1, 3))*output
        image_dict[i] = result    

        output = np.ones((size,size,3),dtype=np.uint8)*255
        output_convex = np.zeros((size,size,1),dtype=bool)
    '''
    
    deconvolution = np.expand_dims(imDeconvolved[:, :, 0],axis=2)
    deconvolution = np.tile(deconvolution,(1,1,3))
    for i in range(len(objProps)):

        c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
        width = objProps[i].bbox[3] - objProps[i].bbox[1] 
        height = objProps[i].bbox[2] - objProps[i].bbox[0] 
        if min(int(c[1] - 0.5 * height), int(c[0] - 0.5 * width))<0 or max(int(c[1] - 0.5 * height)+height, int(c[0] - 0.5 * width)+width)>1200:
            continue
        if min(int(c[1] - 32), int(c[0] - 32))<0 or max(int(c[1] - 32)+64, int(c[0] - 32)+ 64)>1200:
            continue
        #img_segment = np.expand_dims(objProps[i].image, axis=2)
        img = imInput[int(c[1] - 0.5 * height):int(c[1] - 0.5 * height)+height, int(c[0] - 0.5 * width):int(c[0] - 0.5 * width)+width,:]
        ''' 
        if max(height,width)>32:
            scale = 32/float(max(height,width))
            height, width = int(height*scale), int(width*scale)
            img = sp.misc.imresize(img, (height, width))
        
        npad = ((16-height/2,32-height-(16-height/2)),(16-width/2,32-width-(16-width/2)),(0,0))
        segmentate_image = np.pad(img, pad_width=npad,constant_values=0,mode='constant')
        '''
        segmentate_image = sp.misc.imresize(img, (32, 32))
        image_dict[i] = segmentate_image
    
    image = np.array(image_dict.values())
    np.save('/disk1/0717_bbox_resize/{0}.npy'.format(inputImageFile[-9:-4]),image)
    print ('done')

def segmentation_0721(image_name):
    inputImageFile = image_name
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    refImageFile = ('/disk1/rpn/data/BM_GRAZ/source/BM_GRAZ_HE_0007_01.png')  # L1.png
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)

    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd,I_0=255 )
    I_0=255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']    
    # Perform color deconvolution
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)
    #print('Estimated stain colors (rows):', w_est.T[:2])
    # Display results
        #print channel
       # plt.imshow(deconv_result.Stains[:, :, channel])
       # _ = plt.title(stains[i], fontsize=titlesize)



    imNucleiStain = deconv_result.Stains[:, :, 1]
    #plt.figure()
    #plt.imshow(imNucleiStain)
    #imNucleiStain =imDeconvolved[:, :, 0]
    foreground_threshold =120

    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < foreground_threshold)



    min_radius = 5
    max_radius = 30
    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))
    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10
    imNucleiSegMask1, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)
    # filter out small objects
    min_nucleus_area = 200
    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask1, min_nucleus_area).astype(np.int)
    # compute nuclei properties
    objProps = skimage.measure.regionprops(imNucleiSegMask)
    print 'Number of nuclei = ', len(objProps)
    
    imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)

    k= (imNucleicompact==-1)
    imNucleicompact1=np.copy(k)
    for ii in range(0,1200):
        for jj in range(0,1200):
            if imNucleicompact[ii,jj]>0:
                imNucleicompact1[ii,jj]=1

    imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)

    imInput2 = np.copy(imInput)
    listt = []

    import cv2
    for i in range(1,imNucleicompact2.max()):

        k =  (imNucleicompact2==i)
        location = np.where(k == 1)
        x_min, y_min = min(location[0]),min(location[1])
        x_max, y_max = max(location[0]),max(location[1])
        space = (x_max-x_min)*(y_max-y_min)

        if space<450 and space>100:
            segmentate = k[x_min:x_max,y_min:y_max]
            segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
            img = imInput[x_min:x_max,y_min:y_max,:]
            img = img*segmentate
            listt.append(img)
            #plt.imshow(img)
            #plt.show()

        if space>449:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7)) 
            k.dtype=np.uint8
            eroded=cv2.erode(k,kernel);
            dilated = cv2.dilate(eroded,kernel)
            new_seg = skimage.measure.label(dilated,connectivity = 1)
            for j in range (1,new_seg.max()+1):
                kk =  (new_seg==j)
                location1 = np.where(kk == 1)
                x_min1, y_min1 = min(location1[0]),min(location1[1])
                x_max1, y_max1 = max(location1[0]),max(location1[1])
                space1 = (x_max1-x_min1)*(y_max1-y_min1)
                if space1< 800:
                    segmentate = kk[x_min1:x_max1,y_min1:y_max1]
                    segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
                    img1 = imInput[x_min1:x_max1,y_min1:y_max1,:]
                    img1 = img1*segmentate
                    listt.append(img1)
                    #plt.imshow(img1)
                    #plt.show()
    image_dict={}
    n = 0
    for img in listt:
        color_mean = img.mean(axis=2)
        for i in range(0, color_mean.shape[0]):
            for j in range(0, color_mean.shape[1]):
                if color_mean[i,j] == 0.0:
                    img[i,j,:] = 255
    
        height, width = img.shape[0], img.shape[1]
        if max(height,width)>32:
            scale = 32/float(max(height,width))
            height, width = int(height*scale), int(width*scale)
            img = sp.misc.imresize(img, (height, width))
        
        npad = ((16-height/2,32-height-(16-height/2)),(16-width/2,32-width-(16-width/2)),(0,0))
        segmentate_image = np.pad(img, pad_width=npad,constant_values=255,mode='constant')
        image_dict[n] = segmentate_image
        n+=1
        
    image = np.array(image_dict.values())
    np.save('/disk1/0721_fullfill/{0}.npy'.format(inputImageFile[-9:-4]),image)
    print ('done')

