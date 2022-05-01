# -*- coding: utf-8 -*-

import numpy as np

class BubbleLosses():
    def __init__(self):
        self.trn_loss_r = 0.0   # Regression loss training
        self.val_loss_r = 0.0   # Regression loss validation
        self.trn_loss_b = 0.0   # Binary classification loss training
        self.val_loss_b = 0.0   # Binary classification loss validation
        self.trn_loss   = 0.0   # Total loss training   
        self.val_loss   = 0.0   # Total loss validation
        
    def update_trn_metrics(self,loss_r,loss_b,loss):
        
        self.trn_loss_r += loss_r.item()
        self.trn_loss_b += loss_b.item()
        self.trn_loss   += loss.item()
        
    def update_val_metrics(self,loss_r,loss_b,loss):
        
        self.val_loss_r += loss_r.item()
        self.val_loss_b += loss_b.item()
        self.val_loss   += loss.item()
        
    def normalize(self,len_trn_dataloader, len_val_dataloader):
        
        self.trn_loss_r = self.trn_loss_r/float(len_trn_dataloader)
        self.trn_loss_b = self.trn_loss_b/float(len_trn_dataloader)
        self.trn_loss  = self.trn_loss/float(len_trn_dataloader)
        
        self.val_loss_r = self.val_loss_r/float(len_val_dataloader)
        self.val_loss_b = self.val_loss_b/float(len_val_dataloader)
        self.val_loss  = self.val_loss/float(len_val_dataloader)
        
        

class LogVars():
    def __init__(self, Nepochs):
        
        # Arrays with training and validation regression loss:
        self.train_losses_r      = np.zeros(Nepochs)
        self.val_losses_r        = np.zeros(Nepochs)
        
        # Arrays with training and validation binary classification loss:
        self.train_losses_b      = np.zeros(Nepochs)
        self.val_losses_b        = np.zeros(Nepochs) 
        
        # Arrays with training and validation total loss:
        self.train_losses        = np.zeros(Nepochs)
        self.val_losses          = np.zeros(Nepochs)
               
    def update(self, epoch, Nepochs, epoch_losses):

        self.train_losses_r[epoch]      = epoch_losses.trn_loss_r
        self.val_losses_r[epoch]        = epoch_losses.val_loss_r
        self.train_losses_b[epoch]      = epoch_losses.trn_loss_b
        self.val_losses_b[epoch]        = epoch_losses.val_loss_b      
        self.train_losses[epoch]        = epoch_losses.trn_loss
        self.val_losses[epoch]          = epoch_losses.val_loss
               
        print('Epoch {}/{} Train loss {} val loss {}'.format(
            epoch, Nepochs, epoch_losses.trn_loss, epoch_losses.val_loss)) 


    def save(self,modeldir):
        
        np.save(modeldir + '/' + 'train_loss_r',        self.train_losses_r)
        np.save(modeldir + '/' + 'val_loss_r',          self.val_losses_r)
        np.save(modeldir + '/' + 'train_loss_b',        self.train_losses_b)
        np.save(modeldir + '/' + 'val_loss_b',          self.val_losses_b)
        np.save(modeldir + '/' + 'train_loss',          self.train_losses)
        np.save(modeldir + '/' + 'val_loss',            self.val_losses)

