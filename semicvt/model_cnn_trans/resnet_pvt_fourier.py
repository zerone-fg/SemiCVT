import torch
import torch.nn as nn
import torch.nn.functional as F
from .PVT import ConvBNReLU, PVTFFN, PVTBlock
from .LGIModule import LGIModule

from .base import get_syncbn

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


model_urls = {
    "resnet18": "/path/to/resnet18.pth",
    "resnet34": "/path/to/resnet34.pth",
    "resnet50": "/apdcephfs/share_1290796/huimin/pretrained_ST/resnet50.pth",
    "resnet101": "/apdcephfs/share_1290796/huimin/pretrained_ST/resnet101.pth",
    "resnet152": "/path/to/resnet152.pth",
}


class ConvMerging(nn.Module):
    r""" Conv Merging Layer.
    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.norm(x)
        # B, C, H, W -> B, H*W, C
        x = self.reduction(x).flatten(2).permute(0, 2, 1)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False],
        sync_bn=False,
        multi_grid=False,
        fpn=False,
    ):
        super(ResNet, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.fpn = fpn
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, self.inplanes),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            multi_grid=multi_grid,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # trans
        self.block_layer = [1, 1, 1]
        self.trans_MHSA = nn.ModuleList()
        self.trans_FFN = nn.ModuleList()
        self.channel_trans = [512, 1024, 2048]
        self.trans_downlayers = nn.ModuleList()

        self.channel_cnn = [128, 256, 512, 1024, 2048]
        self.stride = [2, 1, 1]
        self.down_dim = nn.ModuleList()
        self.up_dim = nn.ModuleList()
        self.mhsa_channels = [256, 256, 256]
        self.sr_ratio = [4, 4, 4]
        self.ftc_interact = nn.ModuleList()

        for i in range(len(self.block_layer)):
            MHSA = []
            FFN = []
            for _ in range(self.block_layer[i]):
                MHSA.append(
                    PVTBlock(
                        in_channels=self.mhsa_channels[i],
                        sr_ratio=self.sr_ratio[i],
                        num_heads=8,
                    )
                )
                FFN.append(
                    PVTFFN(
                        in_channels=self.mhsa_channels[i],
                        R=4
                    )
                )
            self.trans_MHSA.append(nn.Sequential(*MHSA))
            self.trans_FFN.append(nn.Sequential(*FFN))
            self.up_dim.append(ConvBNReLU(self.mhsa_channels[i], self.channel_trans[i], kernel_size=1, padding=0))
            self.down_dim.append(ConvBNReLU(self.channel_trans[i], self.mhsa_channels[i], kernel_size=1, padding=0))
            self.trans_downlayers.append(
                ConvBNReLU(self.channel_cnn[i+1], self.channel_trans[i], kernel_size=1,
                           stride=self.stride[i], padding=0))
            self.ftc_interact.append(LGIModule(dim_cnn=self.channel_cnn[i + 2], dim_trans=self.mhsa_channels[i]))

    def get_outplanes(self):
        return self.inplanes

    def get_auxplanes(self):
        return self.inplanes // 2

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, multi_grid=False
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        grids = [1] * blocks
        if multi_grid:
            grids = [2, 2, 4]

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation * grids[0],
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation * grids[i],
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, isHint=False):
        c0 = self.conv1(x)
        c0 = self.bn1(c0)
        c0 = self.relu(c0)  # [b, 128, 129, 129]

        c0 = self.maxpool(c0)
        c1 = self.layer1(c0)       # [b, 128, 129, 129]

        '''c2 t2'''
        c2 = self.layer2(c1)       # [b, 512, 65, 65]
        t1 = self.trans_downlayers[0](c1)   # [b, 512, 65, 65]
        t1 = self.down_dim[0](t1)
        t2 = self.trans_FFN[0](self.trans_MHSA[0](t1))
        c2, t2 = self.ftc_interact[0](c2, t2)
        t2 = self.up_dim[0](t2)
        t2 = t2 + c2

        '''c3 t3'''
        c3 = self.layer3(c2)       # [b, 512, 65, 65]
        t3 = self.trans_downlayers[1](t2)   # [b, 512, 65, 65]  [b, 512, 33, 3]
        t3 = self.down_dim[1](t3)
        t3 = self.trans_FFN[1](self.trans_MHSA[1](t3))
        c3, t3 = self.ftc_interact[1](c3, t3)
        t3 = self.up_dim[1](t3)
        t3 = t3 + c3

        '''c3 t3'''
        c4 = self.layer4(c3)       # [b, 512, 65, 65]
        t4 = self.trans_downlayers[2](t3)   # [b, 512, 65, 65]
        t4 = self.down_dim[2](t4)
        t4 = self.trans_FFN[2](self.trans_MHSA[2](t4))
        c4, t4 = self.ftc_interact[2](c4, t4)
        t4 = self.up_dim[2](t4)
        t4 = t4 + c4

        if self.training and isHint:
            loss_hint = self.hintLoss4(c4, t4)
            return [c1, c1, c4, t4, loss_hint]
        else:
            return [c1, c1, c4, t4]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_url = model_urls["resnet18"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet34"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet50"]
        saved_state_dict = torch.load(model_url)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet101"]
        saved_state_dict = torch.load(model_url)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet152"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model
