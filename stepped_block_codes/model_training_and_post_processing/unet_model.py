"""
unet_model.py

WHAT IS A U-NET (for this task)
----------------------------------
A U-Net is a CONVOLUTIONAL encoder-decoder with SKIP CONNECTIONS. It
processes data on a REGULAR GRID (unlike MLP which is pointwise, or
GNO/GINO which uses mesh edge_index directly) -- here, a 3D voxel grid
built by grid_utils.py from your unstructured mesh.

  ENCODER (downsampling path): repeatedly applies Conv3d + downsample,
  learning increasingly abstract, larger-receptive-field features.

  DECODER (upsampling path): repeatedly upsamples back to full
  resolution, at each step CONCATENATING the matching encoder layer's
  features via a "skip connection" -- this lets the network combine
  coarse/global context (from deep layers) with fine/local detail
  (from shallow layers), which is exactly what a temperature field
  with both broad cooling trends AND local hotspots needs.

Input:  [batch, 5, D, H, W]  -- occupancy, SDF, pour_temp, mould_temp,
                                  time (all broadcast across the grid
                                  except occupancy/SDF, which vary
                                  spatially -- see grid_utils.py)
Output: [batch, 1, D, H, W]  -- predicted temperature grid

Unlike MLP (pointwise, no notion of neighbors) or GNO (uses exact
mesh edges), U-Net's 3D convolutions give it built-in LOCAL SPATIAL
AWARENESS (each output voxel is influenced by a neighborhood of
input voxels) at the cost of the regridding quantization error
described in grid_utils.py.

Requires: pip install torch
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class TemperatureUNet3D(nn.Module):
    """
    3-level 3D U-Net, sized for small grids (e.g. 32^3) to keep VRAM
    usage reasonable on both a 6GB laptop GPU and Kaggle's T4.
    """
    def __init__(self, in_channels=5, base_channels=16, out_channels=1):
        super().__init__()

        # --- Encoder ---
        self.enc1 = conv_block(in_channels, base_channels)          # -> base
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(base_channels, base_channels * 2)    # -> 2*base
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)  # -> 4*base
        self.pool3 = nn.MaxPool3d(2)

        # --- Bottleneck ---
        self.bottleneck = conv_block(base_channels * 4, base_channels * 8)

        # --- Decoder (with skip connections) ---
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)  # 8 = 4(up)+4(skip)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)