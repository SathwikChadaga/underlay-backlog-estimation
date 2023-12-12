import torch
import numpy as np

class DataProcessor:
    def __init__(self):
        self.x_min = None
        self.x_max = None

        self.y_min = None
        self.y_max = None

    def create_sequences(self, packets_in_flight, look_back=1):
        sequence_length = packets_in_flight.shape[0]
        # append zeros in the beginning
        packets_in_flight = torch.concat((torch.zeros([look_back-1, packets_in_flight.shape[1]]), packets_in_flight), axis=0)
        X = []
        for ii in range(sequence_length):
            X.append(packets_in_flight[ii:ii+look_back, :])
        return torch.stack(X)
    
    def split_train_test(self, x_all, y_all, train_split_ratio = 0.9):
        n_train = np.int0(train_split_ratio*x_all.shape[0])
        x_train, x_test = x_all[:n_train,:], x_all[n_train:,:]
        y_train, y_test = y_all[:n_train,:], y_all[n_train:,:]
        return x_train, y_train, x_test, y_test

    def scale_train(self, x_train, y_train, is_x_sequenced = False):
        if(is_x_sequenced):
            self.x_min = torch.min(x_train[:,-1,:], axis=0).values
            self.x_max = torch.max(x_train[:,-1,:], axis=0).values
        else:
            self.x_min = torch.min(x_train, axis=0).values
            self.x_max = torch.max(x_train, axis=0).values
        
        self.y_min = torch.min(y_train, axis=0).values
        self.y_max = torch.max(y_train, axis=0).values

        x_train = (x_train - self.x_min)/(self.x_max - self.x_min)
        y_train = (y_train - self.y_min)/(self.y_max - self.y_min)

        return x_train, y_train

    def scale_test(self, x_test, y_test):
        x_test = (x_test - self.x_min)/(self.x_max - self.x_min)
        y_test = (y_test - self.y_min)/(self.y_max - self.y_min)
        return x_test, y_test
    
    def inverse_scale(self, x_scaled, y_scaled):
        x_unscaled = x_scaled*(self.x_max - self.x_min) + self.x_min
        y_unscaled = y_scaled*(self.y_max - self.y_min) + self.y_min
        return x_unscaled, y_unscaled

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