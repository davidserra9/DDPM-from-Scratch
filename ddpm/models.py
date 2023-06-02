import torch
import math
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.gelu = nn.GELU()

    def forward(self, x, t):
        t = self.gelu(self.time_mlp(t))             # Time embedding
        t = t[(..., ) + (None, )*2]
        x = self.bnorm1(self.conv1(x))              # First conv
        h = x + t                                   # Concat time embedding
        h = self.bnorm2(self.gelu(self.conv2(h)))   # Second conv
        return self.transform(h)                    # Down or Upsample


class SimpleUnet(nn.Module):
    def __init__(self, im_sz, in_channels: int = 3, dim_mults=(64, 128, 256, 512, 1024), time_emb_dim: int = 32):
        super().__init__()

        self.in_channels = in_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, dim_mults[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(dim_mults[i], dim_mults[i+1], time_emb_dim) for i in range(len(dim_mults)-1)])

        # Upsample
        self.ups = nn.ModuleList([Block(dim_mults[i], dim_mults[i-1], time_emb_dim, up=True) for i in range(len(dim_mults)-1, 0, -1)])

        self.output = nn.Conv2d(dim_mults[0], in_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)                        # Initial time embedding
        x = self.conv0(x)                           # Initial projection
        residual_inputs = []                        # Unet
        for down in self.downs:                     # Downsample
            x = down(x, t)
            residual_inputs.append(x)               # Save residual connection
        for up in self.ups:                         # Upsample
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)   # Concat residual connection
            x = up(x, t)

        return self.output(x)

if __name__ == "__main__":
    model = SimpleUnet(32, in_channels=3)
    print("Num params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(1, 3, 32, 32)
    t = torch.randn(1, 32, 32)
    y = model(x, t)
    print(y.shape)