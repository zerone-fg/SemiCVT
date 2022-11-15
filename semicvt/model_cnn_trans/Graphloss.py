from .GCN_layer import GraphConvolution_C2N, GraphConvolution_C2C
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillC2C2N(nn.Module):
    def __init__(
        self,
        embed_dim_in: int = 256,
        embed_dim_out: int = 256,
    ):
        super().__init__()
        self.pixels = 10
        self.proj = nn.Linear(embed_dim_in, embed_dim_out)

        self.gcn_cn_1 = GraphConvolution_C2N(256, 256)

        self.gcn_cc_1 = GraphConvolution_C2C(256, 256)

        self.mse_loss_cc = nn.MSELoss(reduction='mean')
        self.mse_loss_cc_adj = nn.MSELoss(reduction='mean')

        self.mse_loss_cn = nn.MSELoss(reduction='mean')
        self.mse_loss_cn_adj = nn.MSELoss(reduction='mean')

    def forward(self, prob_c, feats_c, prob_t,  feats_t):
        pred_c = nn.Softmax(dim=1)(prob_c)
        b, num_classes, h, w = pred_c.shape
        _, d, _, _ = feats_c.shape  # (b, d, h, w)
        pred_c = pred_c.view(b, num_classes, -1)

        feats_c = feats_c.view(b, d, -1).transpose(-1, -2)
        feats_c = self.proj(feats_c)
        feats_c = feats_c.transpose(-1, -2).view(b, d, h , w)

        feats_c_1 = feats_c.unsqueeze(1).repeat(1, num_classes, 1, 1, 1)     # (b, num_classes, d, h, w)
        feats_c_1 = feats_c_1.view(b, num_classes, d, -1).transpose(-1, -2)  # (b, num_classes, hw, d)
        centers_c = torch.sum(feats_c_1 * nn.Softmax(dim=-1)(pred_c).unsqueeze(-1), dim=2)  # (b, c, d)
        feats_c_p = nn.AdaptiveAvgPool2d(output_size=(self.pixels, self.pixels))(feats_c) # (b, d, 30, 30)
        feats_c_p = feats_c_p.view(b, d, -1).transpose(-1, -2).reshape(-1, d).repeat(num_classes, 1, 1)  # (c, b*n, d)
        gcn_in_c = torch.cat((feats_c_p, centers_c.transpose(0, 1)), dim=1)  # (c, b*(1+n), d)
        centers_c_p_adj = nn.Softmax(dim=-1)(torch.bmm(gcn_in_c, gcn_in_c.transpose(-1, -2)))  # (c，b*(1+n), b*(1+n))

        abstract_cn_fc_1 = self.gcn_cn_1(gcn_in_c, centers_c_p_adj)
        pooled_cn_fc = F.layer_norm(abstract_cn_fc_1, (256,))

        with torch.no_grad():
            pred_t = nn.Softmax(dim=1)(prob_t)
            pred_t = pred_t.view(b, num_classes, -1)

            feats_t_1 = feats_t.unsqueeze(1).repeat(1, num_classes, 1, 1, 1)
            feats_t_1 = feats_t_1.view(b, num_classes, d, -1).transpose(-1, -2)
            centers_t = torch.sum(feats_t_1 * nn.Softmax(dim=-1)(pred_t).unsqueeze(-1), dim=2)  # (b, c, d)
            feats_t_p = nn.AdaptiveAvgPool2d(output_size=(self.pixels, self.pixels))(feats_t) # (b, d, 30, 30)
            feats_t_p = feats_t_p.view(b, d, -1).transpose(-1, -2).reshape(-1, d).repeat(num_classes, 1, 1)  # (c, b*n, d)
            gcn_in_t = torch.cat((feats_t_p, centers_t.transpose(0, 1)), dim=1)  # (c, b*(n+1), d)
            centers_t_p_adj = nn.Softmax(dim=-1)(torch.bmm(gcn_in_t, gcn_in_t.transpose(-1, -2)))

            abstract_cn_ft_1 = self.gcn_cn_1(gcn_in_t, centers_t_p_adj)
            pooled_cn_ft = F.layer_norm(abstract_cn_ft_1, (256,))    # [21, 1, 256]

        # C-N
        loss_contrast_cn = self.mse_loss_cn(pooled_cn_fc, pooled_cn_ft)

        # C-C
        centers_c_rf = abstract_cn_fc_1[:, b * (self.pixels * self.pixels):, :].reshape(-1, d)         #  c, b, d
        centers_c_c_adj = nn.Softmax(dim=-1)(centers_c_rf @ centers_c_rf.transpose(-1, -2))   # (b*cls+n, b*cls+n)
        abstract_cc_fc_1 = self.gcn_cc_1(centers_c_rf, centers_c_c_adj)
        pooled_cc_fc = F.layer_norm(abstract_cc_fc_1, (256,))

        with torch.no_grad():
            centers_t_rf = abstract_cn_ft_1[:, b * (self.pixels * self.pixels):, :].reshape(-1, d)         #  c, b, d
            centers_t_c_adj = nn.Softmax(dim=-1)(centers_t_rf @ centers_t_rf.transpose(-1, -2))   # (b*cls+n, b*cls+n)
            abstract_cc_ft_1 = self.gcn_cc_1(centers_t_rf, centers_t_c_adj)
            pooled_cc_ft = F.layer_norm(abstract_cc_ft_1, (256,))

        loss_contrast_cc = self.mse_loss_cc(pooled_cc_fc, pooled_cc_ft)
        return loss_contrast_cn + loss_contrast_cc




















