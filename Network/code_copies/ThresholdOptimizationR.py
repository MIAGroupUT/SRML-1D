# On a trained model, compare F1, precision, and recall for a range of spatial
# tolerances and detection thresholds.

# Import packages:
import torch
from os import listdir
import numpy as np
import platform

from bubblenetwork import DilatedCNN
from bubblenetwork import RandomModel
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats

# Check if running on server:
if platform.system()!='Linux':
    raise NameError('Running on wrong system. Modify BATCH_SIZE and NDATA')

#%% FILE DIRECTORIES
datadir = '/home/blankenn/BubbleSimulations/DATA_FINAL/TRAINING'
filelist = listdir(datadir)

# Investigate this model:
modeldir = '/home/blankenn/BubbleSimulations/Network_FINAL/modelR'

# Investigate the random model:
randommodel = True

# Store the results in this directory:
savedir = modeldir

#%% DATA SET PARAMETERS
NDATA = 1024        # Number of data files
BATCH_SIZE = 64     # Batch size

#%% LOAD THE DATASET
ind = np.arange(0,NDATA,1)   # File indices validation data
    
dataset = load_dataset(datadir,filelist,ind)

data_loader = torch.utils.data.DataLoader(
    dataset,   batch_size=BATCH_SIZE, shuffle=False)


#%% LOAD THE MODEL
epoch = 1249

if randommodel:
    model = RandomModel()

else:   
    model = DilatedCNN(hidden_size=64, depth=12)  
    modelfile = 'epoch_' + str(epoch)
    modelpath = modeldir + '/' + modelfile
    model.load_state_dict(torch.load(modelpath))
    
model = model.cuda()
model.eval()


#%% COMPUTE THE F1 scores
th_list1  = np.arange(-5.0,-1.0,0.5)
th_list1  = np.power(10,th_list1)
th_list2  = np.arange(0.1,0.95,0.05)

# List of thresholds
th_list = np.concatenate((th_list1, th_list2))  

# List of tolerances     
tol_list = np.arange(0,10)

# Matrix with F1 scores for each tolerance and each threshold:
F1_matrix = np.zeros((len(tol_list),len(th_list)))

# For each tolerance and threshold, compute the average F1 score on 
# the dataset:
  
for i, tolerance in enumerate(tol_list): 
    print('tolerance %d' % tolerance) 

    for j,threshold in enumerate(th_list):
        
        F1        = np.array([])
        
        for it, sample_batched in enumerate(data_loader):

            V   = sample_batched['x'].cuda()    # RF signals
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            z = model(V)                        # Predicted bubble distribution
            
            # Compute statistics on the batch:
            TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
            
            P[P==0] = 1e-3     # Prevent divide by 0
            R[R==0] = 1e-3     # Prevent divide by 0

            # Collect F1 score each prediction in a list:
            F1        = np.append(F1,2*P*R/(P+R))

        F1_matrix[i,j] = np.mean(F1)

# List with optimal thresholds for each tolerance:
th_opt = th_list[np.argmax(F1_matrix,1)]

np.save(savedir + '/thresholds_optimal',    th_opt)
np.save(savedir + '/tolerance_list',        tol_list)
