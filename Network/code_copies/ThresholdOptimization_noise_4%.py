# Import packages:
import torch
from os import listdir
import numpy as np
import platform
from scipy import signal

from bubblenetwork import DilatedCNN
from bubblenetwork import RandomModel
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats

# Check if running on server:
if platform.system()!='Linux':
    raise NameError('Running on wrong system. Modify BATCH_SIZE and NDATA')

#%% FILE DIRECTORIES
datadir = '/home/blankenn/SuperResolutionProjectFINAL (main)/DATA_FINAL/'+\
    'TRAINING'
filelist = listdir(datadir)

# Investigate this model:
modeldir = '/home/blankenn/SuperResolutionProjectREVISION/Network_REVISION/'+\
    'modelHS_001_noise_4%'

# Investigate the random model:
randommodel = False

# Store the results in this directory:
savedir = modeldir

#%% DATA SET PARAMETERS
NDATA = 1024        # Number of data files
BATCH_SIZE = 64     # Batch size

#%% NOISE

V_ref = 9.65                                # Reference value noise
noiselevel_p = 4                            # Noise level percentage
noiselevel   = noiselevel_p*V_ref/100       # Noise level

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

            V = sample_batched['x'].cpu().numpy()    # RF signals
    
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
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
