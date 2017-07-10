from train import *

num_epoch = 30
lrQG = 0.00005
lrQD = 0.00005
cont_lamda_1 = np.asarray([[1,1,1,1,1]],dtype=np.float32)
cont_lamda_2 = np.asarray([[0.1,0.1,0.1,0.2,0.2]],dtype=np.float32)
cont_lamda_3 = np.asarray([[0.1,0.2,0.2,0.5,0.5]],dtype=np.float32)
cont_lamda_4 = np.asarray([[0.2,0.2,0.2,0.5,0.5]],dtype=np.float32)
cont_lamda_5 = np.asarray([[0.1,0.1,0.2,0.2,0.5]],dtype=np.float32)
cont_lamda_6 = np.asarray([[0.2,0.5,0.5,1,1]],dtype=np.float32)

for cont_lamda in [cont_lamda_1, cont_lamda_2, cont_lamda_3, cont_lamda_4, cont_lamda_5, cont_lamda_6]:
    train(lrQG, lrQD, cont_lamda, num_epoch)