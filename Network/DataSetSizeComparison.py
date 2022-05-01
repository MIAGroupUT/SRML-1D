# -*- coding: utf-8 -*-

# In this file, the network performance is compared of models trained on 
# different sizes of the training dataset: model HS_01

import numpy as np
import matplotlib.pyplot as plt

#%% LOAD THE RESULTS
modeldir = 'C:/Users/BlankenN/Documents/SuperResolutionProject/Network_FINAL'
final_epoch = 1249

modeldir128  = modeldir + '/datasetsize/modelHS_01_128train'
modeldir256  = modeldir + '/datasetsize/modelHS_01_256train'
modeldir512  = modeldir + '/datasetsize/modelHS_01_512train'
modeldir1024 = modeldir + '/modelHS_01'
modeldir2048 = modeldir + '/datasetsize/modelHS_01_2048train'
modeldir3000 = modeldir + '/datasetsize/modelHS_01_3000train'

modeldirs = [modeldir128, modeldir256, modeldir512, modeldir1024,
             modeldir2048, modeldir3000]


#%% PLOT THE RESULTS
final_trn_losses = np.zeros(len(modeldirs))
final_val_losses = np.zeros(len(modeldirs))
datasetsizes = [128, 256, 512, 1024, 2048, 3000]
legendstr = ['128','256','512','1024','2048','3000']

plt.figure(figsize=(3.5,3), dpi=150)
# For each dataset size, plot the evolution of the validation loss, and 
# collect the final training and validation loss.
for k, modeldir in enumerate(modeldirs):
    val_loss  = np.load(modeldir  + '/val_loss.npy')
    trn_loss  = np.load(modeldir  + '/train_loss.npy')
    final_val_losses[k] = val_loss[-1]
    final_trn_losses[k] = trn_loss[-1]
    plt.plot(val_loss)
    
plt.ylim([0, 2])
plt.xlim([-10, 1250])
plt.legend(legendstr, title="Training set size:",fontsize=6,title_fontsize=6)
plt.xlabel('epochs',fontsize=7)
plt.ylabel('Total loss',fontsize=7)
plt.title('Validation loss as a function of size train dataset',fontsize=7)
plt.grid()
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=7)
plt.tight_layout()
plt.savefig('DatasetsizeVal.svg')

# Plot the final training and validation loss as a function of training set
# size.
plt.figure(figsize=(3.5,3), dpi=150)
plt.plot(datasetsizes, final_val_losses)
plt.plot(datasetsizes, final_trn_losses)
    
plt.xlabel('number of training data',fontsize=7)
plt.ylabel('Total loss',fontsize=7)
plt.legend(['Training loss', 'Validation loss'],fontsize=7 )
plt.title('Final loss as a function of size train dataset',fontsize=7)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=7)
plt.tight_layout()
plt.grid()
plt.savefig('DatasetsizeFinal.svg')