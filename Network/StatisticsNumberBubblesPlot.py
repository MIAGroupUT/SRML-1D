# Visualisation and inspection of the statistics obtained on the test
# set of DATA_FINAL with StatisticsNumberBubbles.py.
# StatisticsNumberBubbles.py and were executed on the GPU server. The 
# statistics were stored in .npy files, which were copied back to the C drive 
# for futher inspection.
#
# F1 score is given as a function of the number of bubbles, and 
# the transmitted acoustic pressure. Scatter plots with one point per
# prediction.

import numpy as np
import matplotlib.pyplot as plt
from customcolormaps import get_custom_colormap

plt.rcParams['svg.fonttype'] = 'none'

# Load the saved data from StatisticsNumberBubbles.py
savedir  = './Statistics_HS_001'
modeldir = './modelHS_001'
modelname = 'HS_001'

# Get optimal threshold values and the corresponding tolerance list:
th_opt    = np.load(modeldir + '/thresholds_optimal.npy')
tol_list  = np.load(modeldir + '/tolerance_list.npy')

F1_matrix = np.load(savedir + '/F1_matrix.npy')   
NB_matrix = np.load(savedir + '/num_bub_matrix.npy') 
PA_matrix = np.load(savedir + '/press_acc_matrix.npy')  

cmap1 = get_custom_colormap('blue')         # Choose colormap        
cmap2 = get_custom_colormap('red')          # Choose colormap


#%% CREATE PLOTS

for tol in [0,4]:
    tol_idx = np.where(tol_list == tol)  # corresponding index in tolerance list

    PA   = np.squeeze(PA_matrix[  tol_idx,:])   # Acoustic pressures
    NB   = np.squeeze(NB_matrix[  tol_idx,:])   # Number of bubbles
  
    score   = np.squeeze(F1_matrix[  tol_idx,:])
    ylabel_str = 'F1 score'
        
    # Optimal threshold for model and tolerance:
    threshold =   th_opt[tol_idx]
        
    PA_str = 'Transmitted pressure (kPa)'
    NB_str = 'Number of bubbles'
    
    title_str = 'Performance of model '+modelname+' on the validation dataset'
    sub_title_str_1 = 'Localisation tolerance = %d' % tol
    sub_title_str_2 = 'Threshold = %.2g' % threshold
    
    #%% First plot: score vs number of bubbles
    plt.figure(figsize=(3.6,2.4), dpi=150)
    plt.grid()
    plt.scatter(NB,   score,   s = 1.5, marker = 'o', c = PA, cmap = cmap1)
    plt.ylim([0,1])
    plt.xlim([0,1000])
    
    cbar = plt.colorbar()
    cbar.set_label(PA_str,size=8,family='arial',fontweight='bold')

    plt.title(title_str + '\n' + sub_title_str_1 + '\n' + sub_title_str_2,
              fontsize=8,family='arial')
    plt.xlabel(NB_str,fontsize=8,family='arial',fontweight='bold')
    plt.ylabel(ylabel_str,fontsize=8,family='arial',fontweight='bold')
    ax = plt.gca()

    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    
    plt.tight_layout()
    

    plt.savefig('Figure_F1_NumBub_tol_%d.svg' % tol)

    
    #%% Second plot: score vs transmitted acoustic pressure
    plt.figure(figsize=(3.6,2.4), dpi=150)
    plt.grid()
    plt.scatter(PA,   score,   s = 1.5, marker = 'o', c = NB, cmap = cmap2)
    plt.ylim([0,1])
    plt.xlim([0,250])
    
    cbar = plt.colorbar()
    cbar.set_label(NB_str,size=8,family='arial',fontweight='bold')
    plt.title(title_str + '\n' + sub_title_str_1 + '\n' + sub_title_str_2, 
              fontsize=8,family='arial')
    plt.xlabel(PA_str,fontsize=8,family='arial',fontweight='bold')
    plt.ylabel(ylabel_str,fontsize=8,family='arial',fontweight='bold')
    ax = plt.gca()

    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    
    plt.tight_layout()

   
    plt.savefig('Figure_F1_PA_tol_%d.svg' % tol)



