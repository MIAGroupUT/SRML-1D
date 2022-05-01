# -*- coding: utf-8 -*-

# This dataloader is intended for loading RF data directly from the MATLAB 
# simulation output files, without loading the ground truth data. It is used 
# for running a trained model on 2D RF data, before applying delay-and-sum 
# reconstruction.
#
# This file is based on bubbledataloader2.py

import numpy as np
import scipy.io as sio

import torch

def load_matlab_rf_data(filepath,signal_type):
    # Read the RF data from a MATLAB simulation output file (.mat)
    #
    # signal_type can be 'V' or 'p'
    
    # Load the MATLAB simulation output file (.mat):
    data = sio.loadmat(filepath)
    
    # Get the 'RF' struct
    vals = data['RF']
    keys = data['RF'].dtype  

    # Read the voltage signals:        
    for struct_cnt in range(0,vals.shape[1]):
        
        # Search for the voltage or pressure key:
        key_cnt = 0
        for key in keys.names:
            if key == signal_type:
                v = vals[0,struct_cnt][key_cnt]
                
                # Store the RF line in a matrix:
                if struct_cnt == 0:
                    V = v
                else:
                    V = np.concatenate((V,v),axis=0)
                    
            key_cnt += 1
    
    V = np.squeeze(V)
    return V

class BubbleDatasetRF(torch.utils.data.Dataset):

    def __init__(self, RFlines):
        self.RFlines   = RFlines
    
    def __len__(self):
        return self.RFlines.shape[0]
    
    def __getitem__(self, idx):
        sample = {'x':  np.squeeze(self.RFlines[idx, :]).unsqueeze(0)} 
        return sample

def load_dataset_rf(filepath):  
    # Only load the RF data, not the ground truth data
    
    signal_type = 'V'
    V = load_matlab_rf_data(filepath,signal_type)
    
    dataset = BubbleDatasetRF(torch.from_numpy(V).float())
        
       
    return dataset
    





        
        