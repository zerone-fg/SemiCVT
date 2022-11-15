import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
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
        )  # change

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

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        if self.fpn:
            return [x1, x2, x3, x4]
        else:
            return [x3, x4]

    def forward(self, x):
        c0 = self.conv1(x)  # [b, 64, h/2, w/2]
        c0 = self.bn1(c0)
        c0 = self.relu(c0)

        '''c1 t1'''
        t1 = self.proj_first(c0)
        c0 = self.maxpool(c0)  # [b, 64, h/4, w/4]
        # MHSA
        t1_mhsa = self.trans_MHSA[0](t1)
        c1 = self.layer1(c0)
        # FFN
        t1_ffn = self.trans_FFN[0](t1_mhsa)
        # fusion cnn
        att_channel = self.channel_interaction[0](F.adaptive_avg_pool2d(c1, output_size=1))
        t1 = F.sigmoid(att_channel) * t1_ffn + c1

        '''c2 t2'''
        t2 = self.trans_downlayers[0](t1)
        # MHSA
        t2_mhsa = self.trans_MHSA[1](t2)
        c2 = self.layer2(c1)
        # FFN
        t2_ffn = self.trans_FFN[1](t2_mhsa)
        # fusion cnn
        att_channel = self.channel_interaction[1](F.adaptive_avg_pool2d(c2, output_size=1))
        t2 = F.sigmoid(att_channel) * t2_ffn + c2

        '''c3 t3'''
        t3 = self.trans_downlayers[1](t2)
        # MHSA
        t3_mhsa = self.trans_MHSA[2](t3)
        c3 = self.layer3(c2)
        # FFN
        t3_ffn = self.trans_FFN[2](t3_mhsa)
        # fusion cnn
        att_channel = self.channel_interaction[2](F.adaptive_avg_pool2d(c3, output_size=1))
        t3 = F.sigmoid(att_channel) * t3_ffn + c3

        '''c4 t4'''
        t4 = self.trans_downlayers[2](t3)
        # MHSA
        t4_mhsa = self.trans_MHSA[3](t4)
        c4 = self.layer4(c3)
        # FFN
        t4_ffn = self.trans_FFN[3](t4_mhsa)
        # fusion cnn
        att_channel = self.channel_interaction[3](F.adaptive_avg_pool2d(c4, output_size=1))
        t4 = F.sigmoid(att_channel) * t4_ffn + c4

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
