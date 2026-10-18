import torch
import torch.nn as nn

def precompute_rope(head_dim, context_length, base=10000):

    assert head_dim % 2 == 0

    # TODO:
    # create frequencies for head_dim // 2 pairs
    # Create rotation pairs for each position in the context length
    indices = torch.arange(0, head_dim, 2).float()

    # Each feature pair has a different angular frequency.
    # Why different frequencies?
    # Because we want some vector pairs to rotate quickly as token positions change, 
    # and others to rotate slowly. 
    # This gives attention information about positional differences at multiple scales.
    inv_freq = 1.0 / (
        base ** (indices / head_dim)
    )

    # Every possible token position
    positions = torch.arange(context_length).float()

    # Each position gets an angle for each frequency
    angles = positions[:, None] * inv_freq[None, :]

    # Duplicate angles for the two halves
    # Earlier was did only first half, now we need to duplicate for the second half of the head dimension
    angles = torch.cat([angles, angles], dim=-1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):

    # x: [batch, heads, seq_len, head_dim]
    B, H, T, D = x.shape

    # Split each head's features into two halves
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]

    # Create the 90-degree-rotated companion
    rotated = torch.cat([-x2, x1], dim=-1)

    # Select the positions we need
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Apply 2D rotations to feature pairs
    result = x * cos + rotated * sin

    return result.to(dtype=x.dtype)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.context_length = context_length

        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # self.qkv = torch.nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)

        cos, sin = precompute_rope(
                    self.d_head,
                    context_length
                )
        
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        batch_size, seq_length, d_in = x.size()

        # Compute Q, K, V
        q = self.w_query(x).view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_key(x).view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_value(x).view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Compute attention scores
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill_(self.mask[:seq_length, :seq_length] == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores / (self.d_head ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.out_proj(output)

        return output

if __name__ == "__main__":
    x = torch.randn(2, 8, 64)

    attn = MultiHeadSelfAttention(
        d_in=64,
        d_out=64,
        num_heads=4,
        context_length=8,
        dropout=0.0,
    )

    out = attn(x)

    print(out.shape)

    cos, sin = precompute_rope(
        head_dim=64,
        context_length=1024
    )

    x = torch.randn(2, 12, 8, 64)
    y = apply_rope(x, cos, sin)

    print(x.shape, y.shape)

    # Position 0 has angle zero, so rotation is identity
    assert torch.allclose(x[:, :, 0], y[:, :, 0], atol=1e-6)

    # Rotations preserve vector length
    assert torch.allclose(
        x.norm(dim=-1),
        y.norm(dim=-1),
        atol=1e-5
    )
    # torch.Size([2, 8, 64])
