import torch
import torch.nn as nn
from einops import rearrange, repeat
from collections import OrderedDict
import math
from . import vit_seg_configs as configs
from os.path import join as pjoin
import numpy as np
import torch.nn.functional as F
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis


class CNNAttention(nn.Module):
    def __init__(self, dim=3, heads=8, dim_head=64, dropout=0., num_patches=1024):
        """
        CNNAttention模块用于实现卷积注意力机制。

        Args:
            dim (int): 输入特征的维度。
            heads (int, optional): 并行注意力头的数量。默认为8。
            dim_head (int, optional): 每个注意力头的维度。默认为64。
            dropout (float, optional): Dropout的概率。默认为0.。
            num_patches (int, optional): 图像分割的补丁数量。默认为1024。
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).cuda()
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()
        """ 
        这段代码是一个神经网络模型中的一部分。它定义了一个名为to_out的模块，该模块是一个nn.Sequential对象。nn.Sequential是一个顺序容器，它按照给定的顺序依次执行包含的模块。
        在这个例子中，to_out包含了三个模块：
        nn.Conv2d：这是一个二维卷积层，它将输入特征图进行卷积操作。它有几个参数，包括输入通道数（inner_dim）、输出通道数（dim）、卷积核大小（kernel_size）、填充大小（padding）和是否使用偏置（bias）。这个卷积层的作用是将输入特征图从inner_dim维度转换为dim维度。
        nn.BatchNorm2d：这是一个二维批归一化层，它对输入进行批归一化操作。它有一个参数，即输入通道数（dim）。批归一化可以提高模型的训练稳定性和泛化能力。
        nn.ReLU：这是一个激活函数层，它将输入进行ReLU激活操作。ReLU函数将所有负值变为零，保持正值不变。这个激活函数可以增加模型的非线性能力。
        最后，如果project_out为True，则to_out还包含一个nn.Conv2d层，否则为nn.Identity层。nn.Identity层是一个恒等映射，它不对输入进行任何操作。这个条件语句的作用是根据project_out的值决定是否在to_out的末尾添加一个额外的卷积层。
        这段代码的作用是将输入特征图从inner_dim维度转换为dim维度，并对转换后的特征图进行批归一化和ReLU激活操作。如果project_out为True，则还会在最后添加一个额外的卷积层。
        """
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        """
        CNNAttention的前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, channels, height, width)。
            mode (str, optional): 模式，可选值为"train"或其他。默认为"train"。
            smooth (float, optional): 平滑项的值。默认为1e-4。

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 前向传播的结果。
                如果mode为"train"，返回形状为(batch_size, channels, height, width)的张量。
                如果mode不为"train"，返回一个元组，包含形状为(batch_size, channels, height, width)的张量和形状为(batch_size, heads, num_patches, num_patches)的注意力张量。
        """
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2))
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm
        factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2)
        dis = factor[:, None, None]*self.dis[None, :, :]
        dis = torch.exp(-dis)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]
        attn = attn * dis[None, :, :, :] # 5.28第一次调试修改1的结果，在这报错“The size of tensor a (256) must match the size of tensor b (1024) at non-singleton dimension 3”
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class CNNTransformer_record(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=1024):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attmaps.append(amap)
        return x, ftokens, attmaps


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True,):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_batchnorm):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class SFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        return x
        
    
class Convfuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1)

    def forward(self, x):
        x = self.fuse(x)
        return x

#---------------------------------------------- S2_attention part -----------------------------------------
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]# 对于三个不同的分割部分使用不同的空间位移操作，然后通过分割注意力操作合并这些移动的部分，生成c通道特征图
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)# 然后通过分割注意力操作合并这些移动的部分，生成c通道特征图
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x

class TransUnet_mlp(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, patch_num=32, dim=1024, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1):# 5.28，针对第一次调试修改dim从512-》1024
        super(TransUnet_mlp, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.encoder = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        num_patches = 16*16# 5.28 第一次调试，解决The size of tensor a (256) must match the size of tensor b (1024) at non-singleton dimension 3这个问题的修改，num_patches为进入CNNattention的feature map的空间分辨率h，w

        self.convformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        # 各个编码层和解码层输出的通道数
        encoder_channels = [256, 512, 1024]
        decoder_channels = (512, 256, 128, 64)
        self.decoder1 = Decoder(in_channels=1024, out_channels=512, skip_channels=512)
        self.decoder2 = Decoder(in_channels=512, out_channels=256, skip_channels=256)
        self.decoder3 = Decoder(in_channels=256, out_channels=128, skip_channels=64)
        self.decoder4 = Decoder(in_channels=128, out_channels=64, skip_channels=0)
        self.s2att = S2Attention(channels=1024)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x, return_features=False):
        # 输入x[16，3，256，256]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将单通道图像扩充为三通道
        x, features = self.encoder(x)
        # encoder之后x[16,1024,16,16]

        # 使用s2_mlp模块处理特征图;同时在encoder之后和convformer之后使用s2mlp
        x = self.s2att(x)

        x= self.convformer(x)  # (B, n_patch, hidden)# 5.28 第一次调试问题1：convformer输入需要512通道数，encoder出来有1024个通道数
        # 尝试将S2mlp放在covformer之后
        x = self.s2att(x)# [16, 1024, 16, 16]

        high_level_feat = x
        
        # 5.28 调试问题2：convformer输出与decoder输入不匹配
        # s2 convformer不改变x的shape
        x = self.decoder1(x, features[0])# ([16, 512, 32, 32])
        
        x = self.decoder2(x, features[1])# ([16, 256, 64, 64])

        x = self.decoder3(x, features[2])# [16, 128, 128, 128]

        x = self.decoder4(x) # [16, 64, 256, 256]

        logits = self.segmentation_head(x)# 5.29 调试问题3：分割头与decoder输出通道数不匹配，修改config['decoder_channels']
        
        if return_features:
            return logits, high_level_feat, features
        return logits
    
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.encoder.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.encoder.root.gn.weight.copy_(gn_weight)
            self.encoder.root.gn.bias.copy_(gn_bias)
            for bname, block in self.encoder.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)
            # if self.transformer.embeddings.hybrid:
            #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
            #         for uname, unit in block.named_children():
            #             unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}