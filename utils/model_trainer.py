# Class to train the model

import numpy as np
import torch

class modelTrainer():
    def __init__(self, criterion, device):
        super(modelTrainer, self).__init__()

        self.device    = device
        self.criterion = criterion
    
    def batch_step(self, model, x_batch, y_batch, optimizer):
        model.train()
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward 
        y_pred = model(x_batch)

        # backward
        loss = self.criterion(y_pred, y_batch)
        loss.backward()

        # optimize
        optimizer.step()
        
        return loss.detach().item()
    
    def batch_step_transformer(self, model, src_batch, tgt_batch, y_batch, optimizer):
        model.train()
        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward 
        y_pred = model(src_batch, tgt_batch)

        # backward
        loss = self.criterion(y_pred, y_batch)
        loss.backward()

        # optimize
        optimizer.step()
        
        return loss.detach().item()

    def get_train_batch(self, x_train, y_train, batch_size = None):
        batch_indices = np.random.choice(np.arange(x_train.shape[0]), batch_size, replace=False)
        return x_train[batch_indices,:], y_train[batch_indices,:]

    def get_train_batch_transformer(self, src_train, tgt_train, y_train, batch_size = None):
        batch_indices = np.random.choice(np.arange(src_train.shape[0]), batch_size, replace=False)
        return src_train[batch_indices,:], tgt_train[batch_indices,:], y_train[batch_indices,:]


