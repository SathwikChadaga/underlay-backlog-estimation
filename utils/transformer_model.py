import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, look_back, num_heads, num_encoder_layers, num_decoder_layers, dropout):
        super(TransformerModel, self).__init__()
        
        self.model_dim = model_dim
        self.positional_encoder = PositionalEncoding(d_model = model_dim, dropout = dropout, max_len = look_back)
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model = model_dim, 
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout)
        self.input_layer = nn.Linear(input_dim, model_dim)
        self.out = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = self.embedding(src) * math.sqrt(self.model_dim)
        # tgt = self.embedding(tgt) * math.sqrt(self.model_dim)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        src = self.input_layer(src)
        tgt = self.input_layer(tgt)
        
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out[-1, :])

        return out

    def evaluate(self, src, tgt):
        with torch.no_grad():
            return self.forward(src, tgt)
        
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