import histomicstk as htk
import numpy as np
import scipy as sp
import skimage.io
import skimage.measure
import skimage.color

def segmentation(imagename):
    inputImageFile = imagename

    imInput = skimage.io.imread(inputImageFile)[:, :, :3]

    refImageFile = ('/disk2/dataset_0724_for_pytorch/1/935.jpg')  # L1.png

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


    s=[]
    # Display results
    for i in 0, 1:
        # Unlike SNMF, we're not guaranteed the order of the different stains.
        # find_stain_index guesses which one we want
        channel = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map[stains[i]], w_est)

        s.append(channel)
        #plt.imshow(deconv_result.Stains[:, :, i])
        #_ = plt.title(stains[i], fontsize=titlesize)

    imNucleiStain = deconv_result.Stains[:, :, s[0]]
    #plt.figure()
    #plt.imshow(imNucleiStain)
    #imNucleiStain =imDeconvolved[:, :, 0]
    foreground_threshold =130

    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < foreground_threshold)

    imNucleicompact = htk.segmentation.label.compact(imFgndMask, compaction=3)

    min_radius = 5
    max_radius = 30

    imLog = htk.filters.shape.clog(imNucleiStain, imNucleicompact,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10

    imNucleiSegMask1, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)



    # filter out small objects
    min_nucleus_area = 100

    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask1, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(imNucleiSegMask)

    print 'Number of nuclei = ', len(objProps)

    imNucleicompact1 = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)

    k= (imNucleicompact1==-1)
    imNucleicompact2=np.copy(k)
    for ii in range(0,k.shape[0]):
        for jj in range(0,k.shape[1]):
            if imNucleicompact1[ii,jj]>0:
                imNucleicompact2[ii,jj]=1
    imNucleicompact2.dtype=np.uint8   

    imNucleicompact2_dilate_xor = htk.segmentation.label.dilate_xor(imNucleicompact2, neigh_width=2)

    imInput2 = np.copy(imInput)

    for i in range(0,imInput.shape[0]):
        for j in range(0,imInput.shape[1]):
            if imNucleicompact2_dilate_xor[i,j]>0:
                imInput2[i,j,0] = 0
                imInput2[i,j,1] = 255
                imInput2[i,j,2] = 0

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
                if space1< 800 and space1>100:
                    segmentate = kk[x_min1:x_max1,y_min1:y_max1]
                    segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
                    img1 = imInput[x_min1:x_max1,y_min1:y_max1,:]
                    img1 = img1*segmentate
                    listt.append(img1)
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
    print(image.shape)
    np.save('/disk2/0724_cell/{0}.npy'.format(imagename.split('/')[-1][:-4]),image)
    print ('done')
       