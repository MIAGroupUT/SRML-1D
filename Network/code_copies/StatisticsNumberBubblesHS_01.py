# For a given model, and for given threshold values, compute the F1 score as
# a function as number of bubbles, and as a function of the acoustic pressure.

# Import packages:
import torch
from os import listdir
import numpy as np
import platform

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats

if platform.system()!='Linux':
    raise NameError('Running on wrong system. Modify batch size, Ntrain, and Nval')

#%% FILE DIRECTORIES
datadir = '/home/blankenn/BubbleSimulations/DATA_FINAL/TESTING'
modeldir = './modelHS_01'
savedir  = 'Statistics_HS_01'
filelist = listdir(datadir)


#%% DATA SET PARAMETERS
NDATA = 960         # Number of data files
BATCH_SIZE = 64     # Batch size

#%% LOAD THE DATASET
ind = np.arange(0,NDATA,1)   # File indices validation data
    
dataset = load_dataset(datadir,filelist,ind)

data_loader = torch.utils.data.DataLoader(
    dataset,   batch_size=BATCH_SIZE, shuffle=False)

#%% LOAD THE MODEL
epoch = 1249
model = DilatedCNN(hidden_size=64, depth=12)  
modelpath = modeldir + '/epoch_' + str(epoch)
model.load_state_dict(torch.load(modelpath)) 
model = model.cuda()
model.eval()

#%%
# Get optimal threshold values and the corresponding tolerance list:
th_opt    = np.load(modeldir + '/thresholds_optimal.npy')
tol_list  = np.load(modeldir + '/tolerance_list.npy')

#%% COMPUTE THE STATISTICS
F1_matrix = np.zeros((len(tol_list),len(dataset)))  # F1 score
NB_matrix = np.zeros((len(tol_list),len(dataset)))  # Number of bubbles
PA_matrix = np.zeros((len(tol_list),len(dataset)))  # Acoustic pressure

for i, tolerance in enumerate(tol_list): 
    print('tolerance %d' % tolerance) 
    
    threshold = th_opt[i]
       
    F1        = np.array([])
    NumBub    = np.array([])
    PressAcc  = np.array([])
    
    for it, sample_batched in enumerate(data_loader):

        V = sample_batched['x'].cuda()      # RF signals
        y = sample_batched['y1'].cuda()     # Ground truth
        z = model(V)                        # Prediction
        
        # Get the transmitted acoustic pressure (pressure at transducer):
        p = sample_batched['y2']            
        PA = p[:,0].cpu().numpy()
        
        # Get the number of bubbles:
        NB = np.sum(y.cpu().numpy(),axis=1)
               
        # Compute statistics on the batch:
        TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
               
        P[P==0] = 1e-3     # Prevent divide by 0
        R[R==0] = 1e-3     # Prevent divide by 0

        # Collect statistics from each prediction in a list:
        F1        = np.append(F1,2*P*R/(P+R))
        NumBub    = np.append(NumBub,NB)
        PressAcc  = np.append(PressAcc,PA)

    F1_matrix[i,:] = F1
    NB_matrix[i,:] = NumBub
    PA_matrix[i,:] = PressAcc

np.save(savedir + '/F1_matrix',F1_matrix)   
np.save(savedir + '/num_bub_matrix',NB_matrix) 
np.save(savedir + '/press_acc_matrix',PA_matrix)  
