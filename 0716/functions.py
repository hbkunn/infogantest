from sklearn.datasets import fetch_mldata
import cv2
import numpy as np
import os
from multiprocessing import Pool as ThreadPool

def get_mnist():
    mnist = fetch_mldata('MNIST original',data_home="/home/msragpu/cellwork/test_dataset/")
    np.random.seed(1234) # set seed for deterministic ordering
   #p = np.datacom.permutation(mnist.data.shape[0])
   #X = mnist.data[p]
    X = mnist.data.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (32,32)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 32, 32)) 
    X = np.tile(X, (1, 3, 1, 1))
    p = np.random.permutation(70000)
    X = X[p]
    X_train = X[:60000]
    X_test = X[60000:70000]
    
    return X_train.reshape(60000,3,32,32)

def get_test():
    mnist = fetch_mldata('MNIST original',data_home="/home/msragpu/cellwork/test_dataset/")
    np.random.seed(1234) # set seed for deterministic ordering
   #p = np.datacom.permutation(mnist.data.shape[0])
   #X = mnist.data[p]
    X = mnist.data.reshape((70000, 28, 28))
    Y = mnist.target
    X = np.asarray([cv2.resize(x, (32,32)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 32, 32)) 
    X = np.tile(X, (1, 3, 1, 1))
    p = np.random.permutation(70000)
    X = X[p]
    Y = Y[p]
    X_test = X[60000:70000]
    Y_test = Y[60000:70000]
    
    return X_test.reshape(10000,3,32,32),Y_test

def bonemarrow_cell():
    X = np.load("../data/data.npy")
    img = X
    X = np.asarray([cv2.resize(x, (32,32)) for x in X])
    X = np.asarray([x[:,:,::-1].transpose((2,0,1)) for x in X])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    return X

def segment_cell():
    pool = ThreadPool(8) 
    root_dir = '/disk1/cell_segment_save/'
    npyList = os.listdir(root_dir)
    npyList = [root_dir+n for n in npyList]
    result = pool.map(np.load, npyList)
    result = np.array(result)
    X = np.asarray([x.transpose((2,0,1)) for x in result])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    return result