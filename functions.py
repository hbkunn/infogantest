import numpy as np
import os
import cv2
import csv
import pandas as pd
from pyspark.sql import SQLContext
from pyspark import SparkConf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def save_img(root_dir):
    X_pics = np.zeros((200,3,320,320),dtype=np.float32)
    imageList = os.listdir(root_dir)
    q = 0
    dic = {}
    
    for n in imageList:
        image_name = n
        n = root_dir + n
        img = cv2.imread(n, 3)
        if img is None:
            continue
        image = img.tolist()
        dic[image_name] = image
    return dic

def load_dataframe():
    my_dict = np.load('./my_file.npy') 
    df = pd.DataFrame(my_dict.item().items(),columns=['image_name', 'image_array'])
    return df
	
def show_image(name, array):
    image = cv2.imwrite(('./out.jpg'), array)
    plt.imshow(Image.open('./out.jpg'))
    plt.axis('off')
    plt.title(name, color="black")
    plt.show()
    
def search_by_name(sparkdf,name):
    result = sparkdf
    if [n.image_name for n in result].index(result) != None:
        show_image(name, np.array(result[index].image_array,dtype=int))
    else:
        print ('no existence')
    
def search_by_index(sparkdf,index):
    result = sparkdf[index]
    array, name = result.image_array, result.image_name
    array = np.array(array,dtype=int)
    show_image(name, array)
    
def compute_RGB_mean(sparkdf):
    result = sparkdf
    mean = np.zeros((3,),dtype=np.float32)
    for n in [n.image_array for n in result]:
        array = np.array(n,dtype=int)
        mean += array.mean(axis = 0).mean(axis = 0)/200
    print ("RGB mean value is", mean)
    
def crop_by_index(sparkdf, index, size):
    result = sparkdf[index]
    array, name = result.image_array, result.image_name
    array = np.array(array,dtype=int)
    array_resize = array[160-size/2:160+size/2, 160-size/2:160+size/2, :]
    show_image(name, array_resize)