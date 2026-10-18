import torch
import torch.nn as nn

from attention import MultiHeadSelfAttention

class LayerNorm_g(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Leanable scale parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # Leanable shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift 
        # If we stopped there, LayerNorm would always force features into that standardized form.
        # normalization gives the network stable inputs
        # scale and shift give the network flexibility Forcing everything on same scale would be restrictive

class RMSNorm_g(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Leanable scale parameter

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / norm
        return self.scale * x_norm

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            nn.GELU(),
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)

class SwiGLU_g(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int((8 / 3) * cfg["emb_dim"])
        self.gate_proj = nn.Linear(cfg['emb_dim'], hidden_dim)
        self.value_proj = nn.Linear(cfg['emb_dim'], hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, cfg['emb_dim'])

    def forward(self, x):

        gate = nn.functional.silu(self.gate_proj(x))
        value = self.value_proj(x)

        x = gate * value

        return self.out_proj(x)

class FeedForwardDeep_SwiGLU(nn.Module):
    def __init__(self, cfg, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList([
            SwiGLU_g(cfg)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)

        return x

class FeedForward_Deep(nn.Module):
    def __init__(self, layer_sizes, use_shortcut=False):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[1]),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[1], layer_sizes[2]),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[2], layer_sizes[3]),
                nn.GELU()
            ),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape: 
                x = x + layer_output
            else:
                x = layer_output
        return x


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['num_heads'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
            num_kv_heads=cfg['num_kv_heads']
        )
        # self.ff = FeedForward(cfg)
        self.ff = SwiGLU_g(cfg)
        # self.ff = FeedForwardDeep_SwiGLU(cfg, num_layers=3)
        self.ln1 = RMSNorm_g(cfg['emb_dim'])
        self.ln2 = RMSNorm_g(cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.drop_emb(x) + shortcut
        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.drop_emb(x) + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        # self.position_embedding = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['num_layers'])])
        self.ln_final = RMSNorm_g(cfg['emb_dim'])
        self.output_layer = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.size()
        # token_embeddings = self.token_embedding(x)

        # positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        # x = token_embeddings + self.position_embedding(positions)
        x = self.token_embedding(x)
        x = self.drop_emb(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.output_layer(x)
        return logits

if __name__ == "__main__":
    cfg = {
    "vocab_size": 1000,
    "emb_dim": 64,
    "context_length": 32,
    "num_heads": 4,
    "num_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
    }
    
    model = GPTModel(cfg)
    tokens = torch.randint(0, cfg["vocab_size"], (2, 8))
    targets = torch.randint(0, cfg["vocab_size"], (2, 8))
    
    logits = model(tokens)
    assert logits.shape == (2, 8, 1000)
    assert torch.isfinite(logits).all()
    
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )
    loss.backward()
    
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print("Output shape:", logits.shape)
    print("Loss:", loss.item())
    print("Forward and backward passes OK")

    model.eval()

    x1 = torch.randint(
        0,
        cfg["vocab_size"],
        (1, 8)
    )

    x2 = x1.clone()

    # Change future tokens
    x2[:, 5:] = torch.randint(
        0,
        cfg["vocab_size"],
        (1, 3)
    )
    print(x1)
    print("x1 and x2")
    print(x2)
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)

    assert torch.allclose(
        logits1[:, :5],
        logits2[:, :5],
        atol=1e-5
    )

    print("Causal test passed")