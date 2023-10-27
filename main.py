import torch
import torch.nn as nn

print(torch.__version__)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads == embed_size), "Emmbedings size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.heads_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        values_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embeddings in self.head pieces
        values = values.reshape(N, values_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = queries.reshape(N, query_len, self.heads, self.heads_dim)

        energy = torch.einsum("nqhd, nkhd->nhgk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e18"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.heads_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm1(forward + x))
        return out





