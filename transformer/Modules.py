import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention

    This is the core "attention" math used inside every attention layer.
    For each position in the sequence, build its output as a
    weighted average of all positions' values, where the weights say "how much
    should I pay attention to each other position?".

    Shape notation used below (this module is run on all attention heads at once,
    so the heads are folded into the batch dimension):
        B  = batch_size * n_head   (number of sequences * number of attention heads)
        Tq = number of query time steps (length of the query sequence)
        Tk = number of key/value time steps (length of the key sequence)
        d  = feature size per head (d_k for q/k, d_v for v; here d_k == d_v)
    """

    def __init__(self, temperature):
        super().__init__()
        # temperature = sqrt(d_k). We divide the scores by it so they don't grow
        # huge as the feature size grows (which would make softmax saturate and
        # kill the gradients). It is set by MultiHeadAttention when this is built.
        self.temperature = temperature
        # Softmax over dim=2 = the KEY axis (Tk), so each query's weights over all
        # keys add up to 1.
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # Input sizes:
        #   q : [B, Tq, d]   queries  ("what each position is looking for")
        #   k : [B, Tk, d]   keys     ("what each position offers")
        #   v : [B, Tk, d]   values   ("the content each position contributes")
        #   mask (optional) : [B, Tq, Tk] boolean, True where attention is forbidden
        #                     (e.g. padding positions added to equalize lengths)

        # 1) Similarity scores: dot-product every query against every key.
        #    k.transpose(1, 2) : [B, Tk, d] -> [B, d, Tk]
        #    bmm = batched matrix multiply (one matmul per item in B):
        #    [B, Tq, d] x [B, d, Tk] -> [B, Tq, Tk]
        attn = torch.bmm(q, k.transpose(1, 2))  # [B, Tq, Tk] similarity scores

        # 2) Scale down by sqrt(d_k) to keep the numbers in a stable range.
        attn = attn / self.temperature          # [B, Tq, Tk]

        # 3) Block forbidden positions: set their score to -inf so that after
        #    softmax their weight becomes exactly 0 (they get ignored).
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # [B, Tq, Tk]

        # 4) Turn raw scores into probabilities (each query's row sums to 1).
        attn = self.softmax(attn)                # [B, Tq, Tk] attention weights

        # 5) Weighted sum of the values using those weights:
        #    [B, Tq, Tk] x [B, Tk, d] -> [B, Tq, d]
        output = torch.bmm(attn, v)              # [B, Tq, d] attended features

        # output : [B, Tq, d]   the new feature for each query position
        # attn   : [B, Tq, Tk]  the weight matrix (handy for plotting alignments)
        return output, attn
