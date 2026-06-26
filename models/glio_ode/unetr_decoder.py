import torch
from einops import rearrange
from torch import nn


class _ConvBlock(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UnetrBasicBlock(nn.Module):
    """Two conv blocks at the same resolution. No upsample."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBlock(in_channels, out_channels),
            _ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnetrUpBlock(nn.Module):
    """ConvTranspose3d (stride 2) -> concat with skip -> two conv blocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2,
        )
        self.block = nn.Sequential(
            _ConvBlock(out_channels + skip_channels, out_channels),
            _ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


def _tokens_to_volume(tokens: torch.Tensor, patch_grid) -> torch.Tensor:
    pd, ph, pw = patch_grid
    return rearrange(tokens, "b (d h w) c -> b c d h w", d=pd, h=ph, w=pw)


class UnetrDecoder(nn.Module):
    """Fuses 4 ViT-trunk checkpoints into full-resolution output.

    Stage 0 (coarsest): UnetrBasicBlock on checkpoint[0] at patch-grid resolution.
    Stage 1..3: UnetrUpBlock with checkpoint[i] as the skip.

    Total upsample factor across the 3 UpBlocks is 2^3 = 8 spatially, so
    we additionally need a final ConvTranspose3d to recover the leftover
    factor implied by patch_size beyond 8. For patch_size (4, 4, 4) the
    overall factor is 4 < 8, so we *under*-upsample inside the decoder
    and add an adaptive interpolate at the head. For patch_size (2, 4, 4)
    we under-upsample on the D dim and rely on the head interpolation as
    well. The head conv is applied after we interpolate to the target
    crop shape so the prediction lives at full resolution.
    """

    def __init__(
        self,
        d_model: int,
        decoder_channels,
        out_channels: int,
        patch_size,
    ):
        super().__init__()
        assert len(decoder_channels) == 4, "decoder_channels must have 4 entries"
        self.patch_size = tuple(patch_size)
        c0, c1, c2, c3 = decoder_channels

        # Project each checkpoint from d_model to its decoder-stage channel count.
        self.proj0 = UnetrBasicBlock(d_model, c0)
        self.proj1 = UnetrBasicBlock(d_model, c1)
        self.proj2 = UnetrBasicBlock(d_model, c2)
        self.proj3 = UnetrBasicBlock(d_model, c3)

        # Upsample blocks (each stride 2).
        self.up1 = UnetrUpBlock(in_channels=c0, skip_channels=c1, out_channels=c1)
        self.up2 = UnetrUpBlock(in_channels=c1, skip_channels=c2, out_channels=c2)
        self.up3 = UnetrUpBlock(in_channels=c2, skip_channels=c3, out_channels=c3)

        self.head = nn.Conv3d(c3, out_channels, kernel_size=1)

    def forward(self, states, patch_grid) -> torch.Tensor:
        assert len(states) == 4, "expected 4 ODE checkpoints"
        s0, s1, s2, s3 = [_tokens_to_volume(s, patch_grid) for s in states]

        x0 = self.proj0(s0)
        x1 = self.proj1(s1)
        x2 = self.proj2(s2)
        x3 = self.proj3(s3)

        # Skips must be upsampled to match the running decoder resolution.
        # We use trilinear interpolation to align spatial sizes before each fuse.
        x1_skip = nn.functional.interpolate(
            x1, scale_factor=2, mode="trilinear", align_corners=False,
        )
        y = self.up1(x0, x1_skip)

        x2_skip = nn.functional.interpolate(
            x2, scale_factor=4, mode="trilinear", align_corners=False,
        )
        y = self.up2(y, x2_skip)

        x3_skip = nn.functional.interpolate(
            x3, scale_factor=8, mode="trilinear", align_corners=False,
        )
        y = self.up3(y, x3_skip)

        # Compute the target full-resolution shape from patch_grid * patch_size.
        target = tuple(g * p for g, p in zip(patch_grid, self.patch_size))
        if y.shape[-3:] != target:
            y = nn.functional.interpolate(
                y, size=target, mode="trilinear", align_corners=False,
            )
        return self.head(y)
