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
        
    image = np.array(image_dict.values())
    np.save('/disk1/cell_segment_save/{0}.npy'.format(inputImageFile[-9:-4]),image)
