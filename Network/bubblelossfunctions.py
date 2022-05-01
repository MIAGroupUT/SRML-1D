# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            output: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not output.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {} and {}" .format(
                    output.device, target.device))
   
        intersection = torch.sum(output * target, dim=1)
        cardinality = torch.sum(output + target, dim=1)
        dice_score = 2. * intersection / (cardinality + self.eps)

        return torch.mean(torch.tensor(1.) - dice_score)
    
def gaussian(N,A,a):
    i = np.arange(1-N,N)
    y = A*np.exp(-a*i**2)
    return y 

criterion_r = torch.nn.L1Loss()     # Regression loss
criterion_d = DiceLoss()            # Dice loss
    
    
def dual_loss(z,y,epsilon1,epsilon2,a):
    
    A = 1               # Amplitude gaussian filter
    M = 30              # Gaussian kernel size: 2M + 1 
    
    
    G = gaussian(M,A,a) # Convolution kernel
    G = torch.Tensor([[G]]).cuda()
               
    # Apply soft label kernel to output 1 and to ground truth 1
    z_soft = F.conv1d(z, G, padding=M-1)
    y_soft = F.conv1d(y.unsqueeze(dim=1), G, padding=M-1)
    
    # Generate discrete labels
    z_clip = torch.relu(z) - torch.relu(z-1)
    y_clip = torch.relu(y) - torch.relu(y-1)

    # Reduce dimensionality
    z      = torch.squeeze(z)
    z_soft = torch.squeeze(z_soft)
    y_soft = torch.squeeze(y_soft)
    z_clip = torch.squeeze(z_clip)
    
    # Compute loss functions
    loss_r = criterion_r(z_soft, y_soft)        # Regression loss
    loss_b = criterion_d(z_clip, y_clip)        # Binary loss  
    
    # Add loss functions
    loss = epsilon1*loss_r + epsilon2*loss_b     # Dual loss
    
    return loss_r, loss_b, loss

    