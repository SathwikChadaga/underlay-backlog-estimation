import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_dim, tgt_dim, model_dim, output_dim, look_back, num_heads, num_encoder_layers, num_decoder_layers, dropout, initial_fillers, device, generate_square_subsequent_mask):
        super(TransformerModel, self).__init__()

        self.generate_square_subsequent_mask = generate_square_subsequent_mask

        self.look_back = look_back
        self.device = device
        self.model_dim = model_dim
        self.tgt_dim = tgt_dim
        self.initial_fillers = initial_fillers

        self.input_layer_src = nn.Linear(src_dim, model_dim) # nn.ModuleList([nn.Linear(src_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)])
        self.input_layer_tgt = nn.Linear(tgt_dim, model_dim) # nn.ModuleList([nn.Linear(tgt_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)])

        self.positional_encoder = PositionalEncoding(d_model = model_dim, dropout = dropout, max_len = look_back)
        
        self.transformer = nn.Transformer(
            d_model = model_dim, 
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout,
            batch_first = True)
        
        self.output_layer = nn.ModuleList([nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, output_dim)]) # nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src = packets in flight of curreent time step and the previous look_back time steps
        # expected src dimension = (batch_size, look_back, num_tunnels)
        
        # tgt = true queue backlogs of the previous look_back time steps
        # expected tgt dimension = (batch_size, look_back-1, num_tunnels)

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
        transformer_out = self.transformer(src, tgt, src_mask, tgt_mask)
        # transformer_out = (transformer_out)

        # if training return all outputs
        for layer in self.output_layer:
            transformer_out = layer(transformer_out)
        # transformer_out = self.output_layer(transformer_out)

        transformer_out = (nn.Tanh()(transformer_out)+1)/2

        return transformer_out
        
    def evaluate(self, src):
        with torch.no_grad():
            src_mask = (torch.zeros(src.shape[1], src.shape[1])).type(torch.bool).to(self.device)
            memory = self.encode(src, src_mask).to(self.device)

            tgt_in = (torch.zeros([src.shape[0], 1, self.tgt_dim]) + self.initial_fillers).to(self.device)
            for ii in range(self.look_back):
                tgt_mask = (self.generate_square_subsequent_mask(tgt_in.size(1), self.device).type(torch.bool))
                tgt_out = self.decode(tgt_in, memory, tgt_mask)[:,-1,:]
                for layer in self.output_layer:
                    tgt_out = layer(tgt_out)
                tgt_out = (nn.Tanh()(tgt_out)+1)/2
                tgt_in = torch.concat((tgt_in, tgt_out.unsqueeze(dim=1)), dim=1)
        return tgt_in[:,1:,:]
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoder(
                            self.input_layer_src(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(
                          self.input_layer_tgt(tgt)), memory, tgt_mask)
        
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
        x = x + self.pe[:x.size(1)].permute(1,0,2)
        return self.dropout(x)