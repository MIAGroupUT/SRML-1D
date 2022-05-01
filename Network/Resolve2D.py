# -*- coding: utf-8 -*-

# This code applies trained model HS_001 to 2D  RF data. This code applies the
# super-resolution neural network per batch of RF lines. All RF lines are
# concatenated again into one matrix per file.
#
# The results are stored in Results15_sr

# Import packages:
import torch
import os
import numpy as np

from bubbledataloadermatlab import load_dataset_rf
from bubblenetwork import DilatedCNN

#%% FILE DIRECTORIES
# Directory where the model parameters are stored:
modeldir = './modelHS_001' 

# Directory where the simulated RF data is stored: 
filedir = '../DelayAndSumFINAL'
filename = 'RFDATA2D00001.mat'

# Directory where the super-resolved RF data will be stored:
destdir = filedir


#%% LOAD THE DATA
BATCH_SIZE = 16     # Batch size

filepath = os.path.join(filedir,filename)

dataset = load_dataset_rf(filepath) 
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size = BATCH_SIZE, shuffle=False)

#%% LOAD THE MODEL
epoch = 1249
model = DilatedCNN(hidden_size=64, depth=12)  
modelfile = 'epoch_' + str(epoch)
modelpath = modeldir + '/' + modelfile
model.load_state_dict(torch.load(modelpath))
model = model.cuda()
model.eval()
    
for it, sample_batched in enumerate(dataloader):
  
    V = sample_batched['x'].cuda()      # RF data
   
    # Apply the model:
    z = model(V)                        # Predicted bubble distribution
           
    # Convert to numpy array:  
    z = np.squeeze(np.transpose(z.cpu().detach().numpy()))
    
    print(z.shape)
    
    # Store the results in a large matrix:
    if it == 0:
        Z = z
    else:
        Z = np.concatenate((Z,z),axis=1)
        
print(Z.shape)


print('Saving' + filename)
# Write the result to a text file:
destfiledir = os.path.join(destdir,filename[0:-4] + '_sr.txt')
with open(destfiledir,'w') as f:
    np.savetxt(f,Z,'%.10f',delimiter=',')
    

    

    
    

    
