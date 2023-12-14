import torch
import numpy as np

class DataProcessor:
    def __init__(self): return

    def create_sequences(self, packets_in_flight, look_back=1):
        sequence_length = packets_in_flight.shape[0]
        # append zeros in the beginning
        packets_in_flight = torch.concat((torch.zeros([look_back-1, packets_in_flight.shape[1]]), packets_in_flight), axis=0)
        X = []
        for ii in range(sequence_length):
            X.append(packets_in_flight[ii:ii+look_back, :])
        return torch.stack(X)
    
    def split_train_test(self, src_all, tgt_all, train_split_ratio = 0.9):
        n_train = np.int0(train_split_ratio*src_all.shape[0])
        
        src_train, src_test = src_all[:n_train,:], src_all[n_train:,:]
        tgt_train, tgt_test = tgt_all[:n_train,:], tgt_all[n_train:,:]

        return src_train, tgt_train, src_test, tgt_test

    def scale_train(self, src_train, tgt_train, is_x_sequenced = True):
        self.src_min = torch.min(src_train[:,-1,:], axis=0).values
        self.src_max = torch.max(src_train[:,-1,:], axis=0).values
        
        self.tgt_min = torch.min(tgt_train[:,-1,:], axis=0).values
        self.tgt_max = torch.max(tgt_train[:,-1,:], axis=0).values

        src_train = (src_train - self.src_min)/(self.src_max - self.src_min)
        tgt_train = (tgt_train - self.tgt_min)/(self.tgt_max - self.tgt_min)

        return src_train, tgt_train

    def scale_test(self, src_test, tgt_test):
        src_test = (src_test - self.src_min)/(self.src_max - self.src_min)
        tgt_test = (tgt_test - self.tgt_min)/(self.tgt_max - self.tgt_min)
        return src_test, tgt_test
    
    def inverse_scale(self, src_scaled, tgt_scaled):
        src_unscaled = src_scaled*(self.src_max - self.src_min) + self.src_min
        tgt_unscaled = tgt_scaled*(self.tgt_max - self.tgt_min) + self.tgt_min
        return src_unscaled, tgt_unscaled

    def create_tgt_input_outputs(self, tgt, initial_fillers):
        tgt_input = torch.zeros_like(tgt) + initial_fillers
        tgt_input[:,1:,:] = tgt[:,:-1,:]
        tgt_output = tgt
        return tgt_input, tgt_output
    
    def feature_transform(self, device, x):
        num_tunnels = x.shape[-1]

        x_transformed = torch.zeros([x.shape[0], x.shape[1], 2*num_tunnels + 1]).to(device)

        x_transformed[:,:,:num_tunnels] = x

        # time differences
        # TODO: use tunnel injection values instead of this difference
        x_transformed[:,1:,num_tunnels:2*num_tunnels] = x[:,1:,:] - x[:,:-1,:]
        x_transformed[:,0,num_tunnels:2*num_tunnels] = x[:,0,:]

        # sum of features
        x_transformed[:,:,-1] = x[:,:,0] + x[:,:,1]

        # difference of features
        # x_transformed[:,:,-2] = x[:,:,0] + x[:,:,1]

        
        # x_transformed = torch.zeros([x.shape[0], x.shape[1], num_tunnels]).to(device)
        # x_transformed[:,1:,:] = x[:,1:,:] - x[:,:-1,:]
        # x_transformed[:,0,:] = x[:,0,:]

        return x_transformed