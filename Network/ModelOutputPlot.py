# -*- coding: utf-8 -*-

import torch
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

from bubbledataloader import load_dataset
from bubblenetwork import DilatedCNN
torch.cuda.empty_cache()

#%% FILE DIRECTORIES
datadir = 'D:/SuperResolutionProject/DATA_FINAL/TESTING'
filelist = listdir(datadir)

modelname = 'HS_01'
modeldir  = './model' + modelname

#%% DATA SET PARAMETERS
NDATA = 10         # Number of data files

#%% LOAD THE DATASET
ind = np.arange(0,NDATA,1)   # File indices data   
dataset = load_dataset(datadir,filelist,ind)
    
    
#%% LOAD THE TRAINED MODEL
epoch = 1249
model = DilatedCNN(hidden_size=64, depth=12)  
modelpath = modeldir + '/epoch_' + str(epoch)
model.load_state_dict(torch.load(modelpath))
    
model = model.cuda()
model.eval()

#%% COMPUTE THE PREDICTION

idx = 2     # File index

V = dataset[idx]['x'].cuda().unsqueeze(0)       # The RF signal
y = dataset[idx]['y1'].cuda().unsqueeze(0)      # The ground truth
p = dataset[idx]['y2'].cpu().unsqueeze(0)       # Pressure data

z = model(V)                                    # Prediction

# Convert signal, ground truth, and prediction to numpy arrays:     
z = torch.squeeze(z)
y = torch.squeeze(y)
z = z.detach().cpu().numpy()
y = y.detach().cpu().numpy()
V  = torch.squeeze(V)
V  = V.detach().cpu().numpy()

PA = p.numpy()[0,0]                             # Acoustic pressure

#%% PLOT THE RESULTS

# Plot the RF signal:
plt.figure(figsize=(3.6,2.0), dpi=150)
plt.plot(V,color=(0.000,0.000,0.475))
plt.xlim([1, 8446])

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=8)

plt.savefig('modelHS_01_output_RF.svg')

##############################################################################
# Plot the ground truth and prediction (zoomed)
Nstart = 3900
Nend = 4500
fig = plt.figure(figsize=(3.6,2.0), dpi=150)
plt.plot(np.arange(Nstart,Nend),  y[Nstart:Nend],color=(0.122,0.467,0.706))
plt.plot(np.arange(Nstart,Nend), -z[Nstart:Nend],color=(1.000,0.322,0.322))

plt.title('model ' + modelname + ', epoch 1249',fontsize=8,family='arial',
          fontweight='bold')
plt.xlabel('grid point',fontsize=8,family='arial',fontweight='bold')
plt.ylim([-1.1,1.1])

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=7)

# Add an axis for time:
Fs = 62.5       # Sampling rate (MHz)
ax1 = fig.axes[0]
ax2 = ax1.twiny()
ax2.set_xlim([Nstart/Fs,Nend/Fs])
ax2.set_xlabel('time (us)',fontsize=8,family='arial')

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=7)


plt.savefig('modelHS_01_output_zoomed.svg')

##############################################################################
# Plot the ground truth and prediction (full scale)
fig = plt.figure(figsize=(3.6,2.0), dpi=150)
plt.plot(y,color=(0.122,0.467,0.706))
plt.plot(-z,color=(1.000,0.322,0.322))

plt.title('model ' + modelname + ', epoch 1249',fontsize=8,family='arial')
plt.xlabel('grid point',fontsize=8,family='arial',fontweight='bold')
plt.ylabel('bubble count',fontsize=8,family='arial',fontweight='bold')
plt.xlim([1, 8446])

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=7)

# Plot boundaries zoomed interval:
plt.plot([Nstart, Nstart],[-1.2, 3.2])
plt.plot([Nend, Nend],[-1.2, 3.2])
plt.ylim([-1.2,3.2])

# Add an axis for time:
ax1 = fig.axes[0]
ax2 = ax1.twiny()
ax2.set_xlim([0,8446/Fs])
ax2.set_xlabel('time (us)',fontsize=8,family='arial')

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=7)

plt.savefig('modelHS_01_output_full.svg')

#%%
print(sum(y))
print(PA)