# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os import listdir

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblelossfunctions import dual_loss
from bubblelogging import LogVars
from bubblelogging import BubbleLosses

torch.cuda.empty_cache()
 
#%% FILE DIRECTORIES
trndir = '/home/blankenn/BubbleSimulations/DATA_FINAL/TRAINING'
valdir = '/home/blankenn/BubbleSimulations/DATA_FINAL/VALIDATION'

trnfilelist = listdir(trndir)
valfilelist = listdir(valdir)

savedir   = '/home/blankenn/BubbleSimulations/Network_FINAL/modelHS_01_2048train'

#%% TRAINING PARAMETERS
NEPOCHS = 1250      # Number of epochs
NTRN = 2048         # Number of training samples
NVAL = 960          # Number of validation samples
BATCH_SIZE = 64     # Batch size


epsilon1 = 1        # Proportionality constant soft loss
epsilon2 = 1.6      # Proportionality constant Dice loss
a = 0.1             # Width paramater gaussian convolution kernel  


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
        
        V   = sample_batched['x'].cuda()    # RF signals
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
        
        V   = sample_batched['x'].cuda()    # RF signals
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
    
