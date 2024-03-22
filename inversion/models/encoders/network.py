import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.transformer import TransformerDecoderLayer
from models.rotary_embedding import RotaryEmbedding

class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir'):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        b = x.shape[0]
        x = self.input_layer(x)
        # print(cam.shape)
        
        # print(cam_emb.shape)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            if i == 23:
                c3 = x
        p2 = self._upsample_add(c3, self.latlayer1(c2)) # residual
        p1 = self._upsample_add(p2, self.latlayer2(c1))

        
        return c3, p2, p1



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            bottleneck_IR(in_channels, mid_channels, 1),
            bottleneck_IR(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    
class Res_encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=512, bilinear=True):
        super(Res_encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc_1 = bottleneck_IR(n_channels, 32, 2)
        self.inc_2 = bottleneck_IR(32, 64, 1)

        self.down1 = (Down(64, 128)) # 32
        self.down2 = (Down(128, 256)) # 16
        self.down3 = (Down(256, 512)) # 8
       
        self.outc2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.outc3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    def forward(self, x):
        x = self.inc_2(self.inc_1(x))
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        p1 = x4
        p2 = self._upsample_add(x4, self.outc2(x3))
        p3 = self._upsample_add(p2, self.outc3(x2))

        return p1, p2, p3



class pos_mapping(Module):
    def __init__(self):
        super(pos_mapping, self).__init__()
        d_model=512
        rotary = RotaryEmbedding(dim=d_model)
        self.transformerlayer_coarse = TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, dropout=0.1, activation=F.gelu, batch_first=True, rotary=rotary)
        self.transformerlayer_medium = TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, dropout=0.1, activation=F.gelu, batch_first=True, rotary=rotary)
        self.transformerlayer_fine = TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, dropout=0.1, activation=F.gelu, batch_first=True, rotary=rotary)

        
    def forward(self, c3, p2, p1, query, cam_emb):
        p1 = p1.flatten(2).permute(0, 2, 1)
        p2 = p2.flatten(2).permute(0, 2, 1)
        c3 = c3.flatten(2).permute(0, 2, 1)
        
        query_coarse = self.transformerlayer_coarse(query, c3, cam_emb)
        query_medium = self.transformerlayer_medium(query_coarse, p2, cam_emb)
        query_fine = self.transformerlayer_fine(query_medium, p1, cam_emb)
        return query_fine