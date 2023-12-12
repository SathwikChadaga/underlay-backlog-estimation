import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_dim, tgt_dim, model_dim, output_dim, look_back, num_heads, num_encoder_layers, num_decoder_layers, dropout, device):
        super(TransformerModel, self).__init__()

        self.look_back = look_back
        self.device = device
        self.model_dim = model_dim
        self.tgt_dim = tgt_dim

        self.input_layer_src = nn.Linear(src_dim, model_dim) # nn.ModuleList([nn.Linear(src_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)])
        self.input_layer_tgt = nn.Linear(tgt_dim, model_dim) # nn.ModuleList([nn.Linear(tgt_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)])

        self.positional_encoder = PositionalEncoding(d_model = model_dim, dropout = dropout, max_len = look_back)
        
        self.transformer = nn.Transformer(
            d_model = model_dim, 
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout)
        
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        # src = packets in flight of curreent time step and the previous look_back time steps
        # expected src dimension = (batch_size, look_back, num_tunnels)
        
        # tgt = true queue backlogs of the previous look_back time steps
        # expected tgt dimension = (batch_size, look_back-1, num_tunnels)

        # reshape to fit transformers shape requirements
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # convert to transformers dimension
        # for layer in self.input_layer_src:
        #     src = layer(src)
            
        # for layer in self.input_layer_tgt:
        #     tgt = layer(tgt)
        
        src = self.input_layer_src(src)
        tgt = self.input_layer_tgt(tgt)
        
        # add poistional encoding
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # get transformer output
        transformer_out = self.transformer(src, tgt)
        # transformer_out = (transformer_out)

        # if training return all outputs
        out = self.output_layer(transformer_out)
        out = out.permute(1,0,2)
            
        return out
        
    def evaluate(self, src):
        with torch.no_grad():
            zeros_pads = torch.zeros([src.shape[0], 1, self.tgt_dim]).to(self.device)
            tgt_in = zeros_pads.clone()
            for ii in range(self.look_back):
                tgt_out = self.forward(src, tgt_in)
                tgt_in = torch.concat((zeros_pads, tgt_out), dim=1)
        return tgt_out[:,-1,:]
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)