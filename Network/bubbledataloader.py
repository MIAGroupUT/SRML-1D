# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

def load_data(filedir,filelist,ind):
    """
    Load the bubble count data, the RF data, and the pressure field data into 
    separate 2D arrays.
    """

    count = 0
    
    for n in ind:
        
        filename = filelist[n]
        bubbles  = []           # Bubble count array
        RFline   = []           # RF array
        pfield   = []           # Pressure field array
    
        # Read the text file line by line:
        with open(os.path.join(filedir,filename)) as f:
            
            for k,line in enumerate(f.readlines()):
                if k>3:
                    values = line.split(',')            
                    bubbles.append(float(values[0]))
                    RFline.append(float(values[1]))
                    pfield.append(float(values[2])/1e3) # kPa           
    
        # Add the 1D arrays to the 2D array. For the first row, create the 2D
        # array.
        if count == 0:
            bubbleData   = np.array([bubbles])
            RFlinesData = np.array([RFline])
            pfieldData  = np.array([pfield])
            count = count + 1
        else:
            bubbleData   = np.concatenate((bubbleData,[bubbles]))
            RFlinesData = np.concatenate((RFlinesData,[RFline]))
            pfieldData  = np.concatenate((pfieldData,[pfield]))
            
    return RFlinesData, bubbleData, pfieldData


class BubbleDataset(torch.utils.data.Dataset):

    def __init__(self, RFlines, locations, pfields):
        self.RFlines   = RFlines
        self.locations = locations
        self.pfields   = pfields
    
    def __len__(self):
        return self.RFlines.shape[0]
    
    def __getitem__(self, idx):
        sample = {'x':  self.RFlines[idx, :].unsqueeze(0),
                  'y1': self.locations[idx, :],
                  'y2': self.pfields[  idx, :]} 
        return sample

def load_dataset(filedir,filelist,ind):
    
    RFlinesData, bubblesData, pfieldsData = load_data(filedir,filelist,ind)
       
    dataset = BubbleDataset(
        torch.from_numpy(RFlinesData).float(),
        torch.from_numpy(bubblesData).float(),
        torch.from_numpy(pfieldsData).float())
        
    return dataset
