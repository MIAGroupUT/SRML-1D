# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import torch
from scipy import signal

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats

V_ref = 9.65                                # Reference value noise
noise_list_p = np.array([1,2,4,8,16,32])    # Noise level percentage
noise_list   = noise_list_p*V_ref/100       # Noise level

#%% FILE DIRECTORIES
datadir = '/home/blankenn/SuperResolutionProjectFINAL (main)/DATA_FINAL/'+\
    'TESTING'
filelist = listdir(datadir)

#%% DATA SET PARAMETERS
NDATA = 960         # Number of data files
BATCH_SIZE = 8      # Batch size

#%% LOAD THE DATASET
ind = np.arange(0,NDATA,1)   # File indices validation data
    
dataset = load_dataset(datadir,filelist,ind)

data_loader = torch.utils.data.DataLoader(
    dataset,   batch_size=BATCH_SIZE, shuffle=False)

#%% NOISE

# Filter parameters
fs = 62.5   # Sampling frequency (MHz)
n = 4       # Order of the butterworth filter
fc = 1.7*3  # Cut-off frequency (MHz)

# low-pass filter coefficients
filt_b, filt_a = signal.butter(n, 2*fc/fs, 'low')

def add_noise(V,sigma,b,a):
    # Add noise to the signals and convert to torch cuda tensor
    
    mu = 0  # Mean value of the random distribution
    
    # Add noise to the signal and apply low-pass filter
    V_noise = V + np.random.normal(mu,sigma,size = V.shape)
    V_noise_filt  = signal.filtfilt(b,a,V_noise,axis=-1)
    
    # Convert the RF signals to torch cuda tensor
    V_noise_filt = torch.from_numpy(V_noise_filt.copy())
    V_noise_filt = V_noise_filt.type(torch.FloatTensor)
    V_noise_filt = V_noise_filt.cuda()
    
    return V_noise_filt

#%% FOR EACH NOISE LEVEL AND TOLERANCE COMPUTE THE F1 SCORE

for k,noiselevel in enumerate(noise_list):
    print(noiselevel) 
    modeldir  = './modelHS_001_noise_' + str(noise_list_p[k]) + '%'
    th_opt    = np.load(modeldir + '/thresholds_optimal.npy')
    tol_list  = np.load(modeldir + '/tolerance_list.npy')
       
    # LOAD THE MODEL:   
    
    torch.cuda.empty_cache()
    epoch = 1249
 
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

            V = sample_batched['x'].cpu().numpy()    # RF signals
    
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
            y = sample_batched['y1'].cuda()     # Ground truth
            z = model(V)                        # Prediction
            
            # Compute statistics on the batch:
            TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
            
            P[P==0] = 1e-3     # Prevent divide by 0
            R[R==0] = 1e-3     # Prevent divide by 0

            # Collect F1 score each prediction in a list:
            F1        = np.append(F1,2*P*R/(P+R))

        F1_array[i] = np.mean(F1)
    np.save('F1_test_' + str(noise_list_p[k]) + '%',F1_array)   
    plt.plot(F1_array)

plt.xlabel('tolerance')
plt.ylabel('F1 score')
plt.title('Maximum F1 (for optimal threshold)')
plt.legend(noise_list_p)
plt.grid()
plt.ylim([0,1])
plt.xlim([0,9])
plt.savefig('ModelComparison_noise.svg')
