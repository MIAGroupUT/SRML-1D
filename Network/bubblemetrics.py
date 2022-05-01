# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
    
def spatial_tolerant_stats(y,z,tolerance,threshold):   
    """"
    INPUT:
    z: prediction, torch tensor with shape (batch size, 1, num grid points)
    y: ground truth, torch tensor with shape (batch size, num grid points)
    threshold: threshold for counting a bubble
    tolerance: spatial tolerance in number of grid points.
    
    OUTPUT: (numpy arrays, one entry per element from batch)
    TP: true positives 
    FP: false positives 
    FN: false negatives
    P:  precision
    R:  recall
    """
    
    threshold_gt = 0.5            # ground truth threshold
    
    # Make bubble count discrete (bubble or no bubble):
    zd = (z >  threshold).float() # discrete prediction
    yd = (y>threshold_gt).float() # discrete ground truth
    yd = torch.unsqueeze(yd,dim=1)
       
    # Apply tolerance kernel to prediction and ground truth:
    h_tol = torch.ones(1, 1, 2*tolerance+1).cuda()   # tolerance interval convolution kernel

    z_tol = F.conv1d(zd, h_tol, padding=tolerance)   # prediction convolved with tolerance kernel
    y_tol = F.conv1d(yd, h_tol, padding=tolerance)   # ground truth convolved with tolerance kernel

    z_tol = (z_tol>=1)                               # Make discrete
    y_tol = (y_tol>=1)                               # Make discrete
    
    TP_array = yd.bool() & z_tol                     # True positives (array)
    FN_array = yd.bool() & torch.logical_not(z_tol)  # False negatives (array)
    FP_array = zd.bool() & torch.logical_not(y_tol)  # False positives (array)
    
    TP = torch.sum(TP_array.float(),dim=2)           # True positives (count)
    FN = torch.sum(FN_array.float(),dim=2)           # False negatives (count)
    FP = torch.sum(FP_array.float(),dim=2)           # False positives (count)
    
     
    FP[(TP+FP)==0] = 1                               # Prevent division by 0
    FN[(TP+FN)==0] = 1                               # Prevent division by 0
    
    P  = torch.squeeze(TP/(TP+FP))                   # Precision
    R  = torch.squeeze(TP/(TP+FN))                   # Recall
    
    # Convert to numpy:
    TP = TP.cpu().detach().numpy()
    FN = FN.cpu().detach().numpy()
    FP = FP.cpu().detach().numpy()   
    P = P.cpu().detach().numpy()
    R = R.cpu().detach().numpy()
    
    return TP, FN, FP, P, R



