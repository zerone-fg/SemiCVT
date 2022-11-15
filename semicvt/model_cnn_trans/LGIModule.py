import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilterInteract(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inplane = dim // 8
        self.spatial_interaction_cnn = nn.Sequential(
            nn.Conv2d(dim , inplane, kernel_size=1),
            nn.BatchNorm2d(inplane),
            nn.GELU(),
            nn.Conv2d(inplane, 1, kernel_size=1),)

        self.spatial_interaction_trans = nn.Sequential(
            nn.Conv2d(dim, inplane, kernel_size=1),
            nn.BatchNorm2d(inplane),
            nn.GELU(),
            nn.Conv2d(inplane, 1, kernel_size=1),)

    def forward(self, x_cnn, h_c, w_c, x_trans, h_t, w_t):
        # # # FFT
        B, _, c_c = x_cnn.shape
        x_cnn = x_cnn.view(B, h_c, w_c, c_c)
        x_cnn = x_cnn.to(torch.float32)
        x_cnn = torch.fft.rfft2(x_cnn, dim=(1, 2), norm='ortho')

        B, _, c_t = x_trans.shape
        x_trans = x_trans.view(B, h_t, w_t, c_t)
        x_trans = x_trans.to(torch.float32)
        x_trans = torch.fft.rfft2(x_trans, dim=(1, 2), norm='ortho')

        x_trans = x_trans.permute(0, 3, 1, 2)
        x_cnn = x_cnn.permute(0, 3, 1, 2)

        att_spatial_cnn = self.spatial_interaction_cnn(x_cnn.real)
        att_spatial_trans = self.spatial_interaction_trans(x_trans.real)

        # # # interact
        x_cnn = F.sigmoid(att_spatial_trans) * x_cnn + x_cnn
        x_trans = F.sigmoid(att_spatial_cnn) * x_trans + x_trans

        x_cnn = x_cnn.permute(0, 2, 3, 1)
        x_trans = x_trans.permute(0, 2, 3, 1)

        # # #IFFT
        x_cnn = torch.fft.irfft2(x_cnn, s=(h_c, w_c), dim=(1, 2), norm='ortho')
        x_trans = torch.fft.irfft2(x_trans, s=(h_t, w_t), dim=(1, 2), norm='ortho')

        x_cnn = x_cnn.reshape(B, h_c * w_c, c_c)
        x_trans = x_trans.reshape(B, h_t * w_t, c_t)

        return x_cnn, x_trans


class LGIModule(nn.Module):
    def __init__(self, dim_cnn, dim_trans, mlp_ratio=4., drop=0., drop_path_cnn=0.25, drop_path_trans=0.25, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        dim = dim_trans // 4
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.cnn_down = nn.Conv2d(dim_cnn, dim, kernel_size=1, padding=0)
        self.cnn_up = nn.Conv2d(dim, dim_cnn, kernel_size=1, padding=0)

        self.norm_cnn_1 = norm_layer(dim)
        self.drop_path_cnn = DropPath(drop_path_cnn) if drop_path_cnn > 0. else nn.Identity()
        self.norm_cnn_2 = norm_layer(dim)
        self.mlp_cnn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.trans_down = nn.Conv2d(dim_trans, dim, kernel_size=1, padding=0)
        self.trans_up = nn.Conv2d(dim, dim_trans, kernel_size=1, padding=0)
        self.norm_trans_1 = norm_layer(dim)
        self.drop_path_trans = DropPath(drop_path_trans) if drop_path_trans > 0. else nn.Identity()
        self.norm_trans_2 = norm_layer(dim)
        self.mlp_trans = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.filter = GlobalFilterInteract(dim)

    def forward(self, x_cnn, x_trans):
        x_cnn_in = x_cnn
        x_trans_in = x_trans
        x_cnn_down = self.cnn_down(x_cnn)
        x_trans_down = self.trans_down(x_trans)
        b, c_c, h_c, w_c = x_cnn_down.shape
        _, c_t, h_t, w_t = x_trans_down.shape

        x_cnn = x_cnn_down.reshape(b, c_c, h_c * w_c).permute(0, 2, 1)
        x_trans = x_trans_down.reshape(b, c_t, h_t * w_t).permute(0, 2, 1)

        x_cnn_out, x_trans_out = self.filter(self.norm_cnn_1(x_cnn), h_c, w_c, self.norm_trans_1(x_trans), h_t, w_t)

        x_trans_out = self.drop_path_trans(self.mlp_trans(self.norm_trans_2(x_trans_out)))
        x_trans_out = x_trans_out.reshape(b, h_t, w_t, c_t).permute(0, 3, 1, 2)

        x_cnn_out = self.drop_path_cnn(self.mlp_cnn(self.norm_cnn_2(x_cnn_out)))
        x_cnn_out = x_cnn_out.reshape(b, h_c, w_c, c_c).permute(0, 3, 1, 2)

        x_trans = x_trans_in + self.trans_up(x_trans_out + x_trans_down)
        x_cnn = x_cnn_in + self.cnn_up(x_cnn_out + x_cnn_down)

        return x_cnn, x_trans
