import torch
import torch.nn as nn
import numpy as np

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    This attention mechanism avoids dimensionality reduction to capture local cross-channel interaction.
    """
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((np.log2(channel) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        out = x * y
        return out.squeeze(-1) if out.dim() == 3 and out.size(-1) == 1 else out

class CoordAttention(nn.Module):
    """
    Coordinate Attention.
    This attention mechanism decomposes channel attention into two 1D feature encoding processes
    that aggregate features along two spatial directions.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool_h = nn.Identity() 
        self.pool_w = nn.AdaptiveAvgPool1d(1)
        
        mid = max(8, channel // reduction)
        self.fc1 = nn.Linear(channel, mid)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(mid, channel)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch, c, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).squeeze(-1)
        
        y = x_h.transpose(1, 2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = y.transpose(1, 2).sigmoid()
        
        x_w = self.fc1(x_w)
        x_w = self.relu(x_w)
        x_w = self.fc2(x_w).sigmoid().unsqueeze(-1)
        
        out = x * y * x_w
        return out.squeeze(-1) if out.dim() == 3 and out.size(-1) == 1 else out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    DropPath (Stochastic Depth).
    This is a regularization technique that randomly drops entire layers during training.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AgeAttnMLP_Optimized(nn.Module):
    """
    An optimized MLP model with attention mechanisms (ECA, CoordAttention),
    DropPath regularization, and residual connections.
    """
    def __init__(self, input_dim=1280, dropout=0.25, drop_path_rate=0.1, se_reduction=16):
        super().__init__()
        
        def make_block(in_f, out_f, dp_rate=0.):
            # Pre-Activation block (Normalization -> Activation -> Linear -> Dropout)
            return nn.Sequential(
                nn.LayerNorm(in_f),
                nn.BatchNorm1d(in_f),
                nn.GELU(),
                nn.Linear(in_f, out_f),
                nn.Dropout(dropout),
                DropPath(dp_rate)
            )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        
        self.block1 = make_block(input_dim, 1024, dp_rates[0])
        self.attn1 = ECA(1024)

        self.block2 = make_block(1024, 768, dp_rates[1])
        self.attn2 = CoordAttention(768)

        self.block3 = make_block(768, 512, dp_rates[2])
        self.attn3 = ECA(512)

        self.block4 = make_block(512, 256, dp_rates[3])
        self.attn4 = CoordAttention(256)

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

        # Residual projections to match dimensions
        self.residual_proj1 = nn.Sequential(nn.Linear(input_dim, 768), nn.GELU()) if input_dim != 768 else nn.Identity()
        self.residual_proj2 = nn.Sequential(nn.Linear(1024, 512), nn.GELU()) if 1024 != 512 else nn.Identity()

    def forward(self, x):
        x1 = self.attn1(self.block1(x))

        x2 = self.attn2(self.block2(x1))
        x2 = x2 + self.residual_proj1(x)

        x3 = self.attn3(self.block3(x2))
        x3 = x3 + self.residual_proj2(x1)

        x4 = self.attn4(self.block4(x3))
        
        out = self.head(x4)
        return out.squeeze(-1)
