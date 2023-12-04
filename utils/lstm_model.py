import torch
import torch.nn as nn

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(LSTMModel, self).__init__()
#         self.hidden_layers = nn.ModuleList()
#         self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         for ii in range(1, len(hidden_sizes)):
#             self.hidden_layers.append(nn.ReLU())
#             self.hidden_layers.append(nn.Linear(hidden_sizes[ii-1], hidden_sizes[ii]))
#         self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = layer(x)
#         x = torch.mean(x, axis=-2)
#         x = nn.ReLU()(self.output_layer(x[:, :]))
#         return x

#     def evaluate(self, x):
#         with torch.no_grad():
#             return self.forward(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for ii in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.LSTM(hidden_sizes[ii-1], hidden_sizes[ii], num_layers=1, batch_first=True))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x, _ = layer(x)
        x = (self.output_layer(x[:, -1, :]))
        # x = nn.ReLU()(self.output_layer(torch.mean(x, axis=-2)))
        return x

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)