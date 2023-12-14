# Class to train the model

import numpy as np
import torch

class modelTrainer():
    def __init__(self, criterion, device):
        super(modelTrainer, self).__init__()

        self.device    = device
        self.criterion = criterion
        self.batch_start_index = 0
        self.random_perm_indices = None
    
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
    
    def batch_step_transformer(self, model, src_batch, tgt_batch_input, tgt_batch_output, src_mask, tgt_mask, optimizer):
        model.train()
        src_batch = src_batch.to(self.device)
        tgt_batch_input = tgt_batch_input.to(self.device)
        tgt_batch_output = tgt_batch_output.to(self.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward 
        y_pred = model(src_batch, tgt_batch_input, src_mask, tgt_mask)
        
        # backward
        loss = self.criterion(y_pred, tgt_batch_output)
        loss.backward()

        # optimize
        optimizer.step()
        
        return loss.detach().item()

    def get_train_batch(self, x_train, y_train, batch_size = None, randomize = False):
        if(randomize): batch_indices = np.random.choice(np.arange(x_train.shape[0]), batch_size, replace=False)
        else:
            batch_indices = (np.arange(batch_size) + self.batch_start_index)%x_train.shape[0]
            batch_indices = self.random_perm_indices[batch_indices]
            self.batch_start_index = (self.batch_start_index + batch_size)%x_train.shape[0]
        return x_train[batch_indices,:], y_train[batch_indices,:]

    def get_train_batch_transformer(self, src_train, tgt_train_input, tgt_train_outpt, batch_size = None, randomize = False):
        if(randomize): batch_indices = np.random.choice(np.arange(src_train.shape[0]), batch_size, replace=False)
        else:
            batch_indices = (np.arange(batch_size) + self.batch_start_index)%src_train.shape[0]
            batch_indices = self.random_perm_indices[batch_indices]
            self.batch_start_index = (self.batch_start_index + batch_size)%src_train.shape[0]
        return src_train[batch_indices,:], tgt_train_input[batch_indices,:], tgt_train_outpt[batch_indices,:]