import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.W_K(embedded)
        Q = self.W_Q(embedded)
        V = self.W_V(embedded)

        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = torch.matmul(Q, torch.transpose(K, 1, 2)) / (self.attention_dim ** 0.5)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        c_length = K.shape[1]
        c_mask = torch.tril(torch.ones(c_length, c_length))
        c_mask = c_mask == 0
        scores = scores.masked_fill(c_mask, float('-inf'))

        # 4. Apply softmax(dim=2) to masked scores
        scores = nn.functional.softmax(scores, dim=2)

        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(torch.matmul(scores, V), decimals=4)
