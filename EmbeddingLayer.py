import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim=512):
        """Embedding layer is simply a matrix
        Each unique token has its own embedding vector which is learnable and chosen for futher calculations
        . """
        super(Embedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, x):
        return self.embedding_matrix[x]


class Positional(nn.Module):
    def __init__(self, dim=512, max_len=5000):
        super(Positional, self).__init__()
        pe = torch.zeros(max_len, dim)

        div_term = torch.zeros(dim // 2) 

        for i in range(d_model // 2):
            div_term[i] = 10000 ** (2 * i / dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x, padding_mask=None):
        x = x + self.pe[:, :x.size(1), :] 
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_dim, max_len, vocab_size=10000):
        super(EmbeddingLayer, self).__init__()
        self.embedding = Embedding(vocab_size, emb_dim)
        self.positional = Positional(emb_dim, max_len)

    def forward(self, x):
        x = self.embedding(x)
        return self.positional(x)

