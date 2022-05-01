# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import torch

from bubblenetwork import DilatedCNN
from bubblenetwork import RandomModel
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats

model_list = ['H', 'HS_001', 'HS_01', 'HS_1', 'S_001', 'S_01', 'S_1', 'R']

#%% FILE DIRECTORIES
datadir = 'D:/SuperResolutionProject/DATA_FINAL/TESTING'
filelist = listdir(datadir)

#%% DATA SET PARAMETERS
NDATA = 960         # Number of data files
BATCH_SIZE = 8      # Batch size

#%% LOAD THE DATASET
ind = np.arange(0,NDATA,1)   # File indices validation data
    
dataset = load_dataset(datadir,filelist,ind)

data_loader = torch.utils.data.DataLoader(
    dataset,   batch_size=BATCH_SIZE, shuffle=False)

#%% FOR EACH MODEL AND TOLERANCE COMPUTE THE F1 SCORE

for k,modelname in enumerate(model_list):
    print(modelname)  
    modeldir  = './model' + modelname
    th_opt    = np.load(modeldir + '/thresholds_optimal.npy')
    tol_list  = np.load(modeldir + '/tolerance_list.npy')
    
    # LOAD THE MODEL:
        
    torch.cuda.empty_cache()
    epoch = 1249
    
    if modelname == 'R':
        model = RandomModel()
    
    else:   
        model = DilatedCNN(hidden_size=64, depth=12)  
        modelpath = modeldir + '/epoch_' + str(epoch)
        model.load_state_dict(torch.load(modelpath))
        
    model = model.cuda()
    model.eval()
    
    # List with F1 scores for each tolerance:
    F1_array = np.zeros(len(tol_list))
  
    for i, tolerance in enumerate(tol_list): 
        print('tolerance %d' % tolerance) 

        threshold = th_opt[i]   	# Optimal threshold for this tolerance
        F1        = np.array([])    # List with F1 scores for each prediction
        
        for it, sample_batched in enumerate(data_loader):

            V   = sample_batched['x'].cuda()    # RF signals
            y = sample_batched['y1'].cuda()     # Ground truth
            z = model(V)                        # Prediction
            
            # Compute statistics on the batch:
            TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
            
            P[P==0] = 1e-3     # Prevent divide by 0
            R[R==0] = 1e-3     # Prevent divide by 0

            # Collect F1 score each prediction in a list:
            F1        = np.append(F1,2*P*R/(P+R))

        F1_array[i] = np.mean(F1)
    np.save('F1_test_' + modelname,F1_array)   
    plt.plot(F1_array)

plt.xlabel('tolerance')
plt.ylabel('F1 score')
plt.title('Maximum F1 (for optimal threshold)')
plt.legend(model_list)
plt.grid()
plt.ylim([0,1])
plt.xlim([0,9])
plt.savefig('ModelComparison.svg')
