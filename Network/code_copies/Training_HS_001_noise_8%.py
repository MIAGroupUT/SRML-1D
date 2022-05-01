# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os import listdir
from scipy import signal

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblelossfunctions import dual_loss
from bubblelogging import LogVars
from bubblelogging import BubbleLosses

torch.cuda.empty_cache()
 
#%% FILE DIRECTORIES
trndir = '/home/blankenn/SuperResolutionProjectFINAL (main)/DATA_FINAL/'+\
    'TRAINING'
valdir = '/home/blankenn/SuperResolutionProjectFINAL (main)/DATA_FINAL/'+\
    'VALIDATION'

trnfilelist = listdir(trndir)
valfilelist = listdir(valdir)

savedir   = '/home/blankenn/SuperResolutionProjectREVISION/'+\
    'Network_REVISION/modelHS_001_noise_8%'

#%% TRAINING PARAMETERS
NEPOCHS = 1250      # Number of epochs
NTRN = 1024         # Number of training samples
NVAL = 960          # Number of validation samples
BATCH_SIZE = 64     # Batch size


epsilon1 = 1        # Proportionality constant soft loss
epsilon2 = 1.6      # Proportionality constant Dice loss
a = 0.01            # Width paramater gaussian convolution kernel  

#%% NOISE

V_ref = 9.65                                # Reference value noise
noiselevel_p = 8                            # Noise level percentage
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


#%% LOAD THE DATASETS
trn_ind = np.arange(0,NTRN,1)   # File indices training data   
val_ind = np.arange(0,NVAL,1)   # File indices validation data
    
trn_dataset = load_dataset(trndir,trnfilelist,trn_ind)  # Training dataset
val_dataset = load_dataset(valdir,valfilelist,val_ind)  # Validation dataset

trn_dataloader = torch.utils.data.DataLoader(
    trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,   batch_size=BATCH_SIZE, shuffle=True)

#%%
# Initialise the network
model = DilatedCNN(hidden_size=64, depth=12)
model = model.cuda()

# Training settings
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Reduce learning rate for last 250 epochs: 
scheduler = StepLR(optimizer, step_size=(NEPOCHS-250), gamma=0.1)

# Preallocate log arrays
logging_variables = LogVars(NEPOCHS)        # Log object for tracking losses

for epoch in range(NEPOCHS):
       
    epoch_losses = BubbleLosses()          # Log object for losses
    
    #%% TRAINING
    model.train()
    optimizer.zero_grad()
    for it, sample_batched in enumerate(trn_dataloader):
        
        V = sample_batched['x'].cpu().numpy()    # RF signals

        # Add noise to the RF signals and convert to torch cuda tensor:
        V = add_noise(V,noiselevel,filt_b,filt_a)
        
        y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
        
        # Forward pass
        z = model(V)                        # Predicted bubble distribution
        
        # Compute loss (regression loss, classification loss, and total loss)    
        loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
        
        # Update training losses log object
        epoch_losses.update_trn_metrics(loss_r,loss_b,loss)
    
        # Backpropagation
        loss.backward()
      
    # Update network parameters:
    optimizer.step()
    
    #%% VALIDATION
    model.eval()
    for it, sample_batched in enumerate(val_dataloader):
        
        V = sample_batched['x'].cpu().numpy()    # RF signals

        # Add noise to the RF signals and convert to torch cuda tensor:
        V = add_noise(V,noiselevel,filt_b,filt_a)
        
        y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
        
        # Forward pass
        z = model(V)                        # Predicted bubble distribution
        
        # Compute loss (regression loss, classification loss, and total loss)       
        loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
        
        # Update validation losses log object
        epoch_losses.update_val_metrics(loss_r,loss_b,loss)
    
    scheduler.step()
    
    #%% LOGGING AND SAVING
    # Divide cumulative losses by length of dataloader
    epoch_losses.normalize(len(trn_dataloader), len(val_dataloader))
    
    # Update logging arrays and print log message  
    logging_variables.update(epoch, NEPOCHS, epoch_losses)
    
    # Save model and logging arrays
    modelfile = 'epoch_' + str(epoch)
    modelpath = savedir + '/' + modelfile
    torch.save(model.state_dict(), modelpath)
    logging_variables.save(savedir)
    
