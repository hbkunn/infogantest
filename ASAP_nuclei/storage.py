from segmentation_process import segmentation
import os 
from multiprocessing import Pool as ThreadPool

pool = ThreadPool(8) 
root_dir = '/disk1/rpn/data/BM_GRAZ/source/'
imageList = os.listdir(root_dir)
imageList = [root_dir+n for n in imageList]
pool.map(segmentation, imageList)