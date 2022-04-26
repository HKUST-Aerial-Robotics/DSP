import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k, attn_dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.d_k

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_x, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_x, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_x, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_x, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_x, bias=False)

        self.attention = ScaledDotProductAttention(d_k=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_x, eps=1e-6)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch, len_x = x.size(0), x.size(1)

        residual = x

        # Pass through the pre-attention projection: b x len_x x (n*d_v)
        # Separate different heads: b x len_x x n x d_v
        q = self.w_qs(x).view(batch, len_x, n_head, d_k)
        k = self.w_ks(x).view(batch, len_x, n_head, d_k)
        v = self.w_vs(x).view(batch, len_x, n_head, d_v)

        # Transpose for attention dot product: b x n x len_x x d_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x len_x x n x d_v
        # Combine the last two dimensions to concatenate all the heads together: b x len_x x (n*d_v)
        out = out.transpose(1, 2).contiguous().view(batch, len_x, -1)
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class MultiHeadAttnEncoderLayer(nn.Module):
    def __init__(self, d_x, d_k, d_v, n_head, d_inner, dropout=0.1):
        super(MultiHeadAttnEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            n_head, d_x, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_x, d_inner, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(
            enc_input, mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn
