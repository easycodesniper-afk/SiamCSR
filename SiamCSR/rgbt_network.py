import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from .config import config
from .attention import GlobalAttentionBlock, CBAM

class SiamCSRNet(nn.Module):
    def __init__(self, ):
        super(SiamCSRNet, self).__init__()

        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)

        self.former_3_layers_featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),  # 0） stride=2
            nn.BatchNorm2d(96),  # 1）
            nn.MaxPool2d(3, stride=2),  # 2） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),  # 6） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),  # 9
            nn.ReLU(inplace=True),
        )

        self.rgb_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.t_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.attn_rgb_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.attn_t_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.attn_rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.attn_t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.template_attention_block = GlobalAttentionBlock()
        self.detection_attention_block = CBAM(512)

    def forward(self, rgb_template, rgb_detection, t_template, t_detection):
        N = rgb_template.size(0)
        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        t_template = self.former_3_layers_featureExtract(t_template)
        t_detection = self.former_3_layers_featureExtract(t_detection)


        rgb_template_feature = self.rgb_featureExtract(rgb_template)  # [bs,256,6,6]
        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)  # [bs,256,24,24]

        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)

        t_template_feature = self.t_featureExtract(t_template)  # [bs,256,6,6]
        t_detection_feature = self.t_featureExtract(t_detection)  # [bs,256,24,24]
        attn_t_template_feature = self.attn_t_featureExtract(t_template)  # [bs,256,6,6]
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)


        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature, attn_t_template_feature)
        union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(union)
        #union = self.detection_attention_block(union)
        #attn_rgb_detection_feature, attn_t_detection_feature = union[:, :256, :, :], union[:, 256:, :, :]


        #===================RGB==================
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                                         4)  # [bs,2*5,256,4,4]
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [bs,4*5,256,4,4]
        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)  # [bs,256,22,22]
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)  # [bs,256,22,22]
        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)  # [1,bsx256,22,22]

        rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)  # [bsx10,256,4,4]
        rgb_pred_score = F.conv2d(rgb_conv_scores, rgb_score_filters, groups=N).reshape(N, 10,
                                                                                        self.score_displacement + 1,
                                                                                        self.score_displacement + 1)
        # bs,10,19,19
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        # bs,256,22,22
        rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)

        rgb_pred_regression = self.rgb_regress_adjust(
            F.conv2d(rgb_conv_reg, rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))

        # ===================ATTN-RGB==================
        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_rgb_conv_score = self.attn_rgb_conv_cls2(attn_rgb_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(attn_rgb_detection_feature)
        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, attn_rgb_score_filters, groups=N).reshape(N, 10,
                                                                                        self.score_displacement + 1,
                                                                                        self.score_displacement + 1)

        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)

        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(
            F.conv2d(attn_rgb_conv_reg, attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))

        # ===================T==================
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)  # bsx10,256,4,4
        t_pred_score = F.conv2d(t_conv_scores, t_score_filters, groups=N).reshape(N, 10,
                                                                                  self.score_displacement + 1,
                                                                                  self.score_displacement + 1)

        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # bs,4*5,256,4,4
        t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)
        t_conv_regression = self.t_conv_r2(t_detection_feature)  # bs,256,22,22
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        t_pred_regression = self.t_regress_adjust(
            F.conv2d(t_conv_reg, t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))
        # ===================ATTN-T==================
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, attn_t_score_filters, groups=N).reshape(N, 10,
                                                                                  self.score_displacement + 1,
                                                                                  self.score_displacement + 1)

        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                      4)
        attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_t_pred_regression = self.attn_t_regress_adjust(
            F.conv2d(attn_t_conv_reg, attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                  self.score_displacement + 1))


        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, \
               rgb_pred_regression + attn_t_pred_regression, attn_rgb_pred_regression + t_pred_regression

    def track_init(self, rgb_template, t_template):
        N = rgb_template.size(0)

        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        t_template = self.former_3_layers_featureExtract(t_template)
        rgb_template_feature = self.rgb_featureExtract(rgb_template)  # 输出 [1, 256, 6, 6]
        t_template_feature = self.t_featureExtract(t_template)  # 输出 [1, 256, 6, 6]

        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        attn_t_template_feature = self.attn_t_featureExtract(t_template)

        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature,
                                                                                             attn_t_template_feature)

        # kernel_score=1,2x5,256,4,4   kernel_regression=1,4x5, 256,4,4
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                                         4)  # [1, 10, 256, 4, 4]
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)  # 2x5, 256, 4, 4
        self.t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [1, 20, 256, 4, 4]
        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [1, 20, 256, 4, 4]
        self.rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)  # 4x5, 256, 4, 4
        self.t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)  # 4x5, 256, 4, 4

        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        self.attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)
        self.attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, rgb_detection, t_detection):
        N = rgb_detection.size(0)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        t_detection = self.former_3_layers_featureExtract(t_detection)

        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)  # 1,256,24,24
        t_detection_feature = self.t_featureExtract(t_detection)

        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)  # 1,256,24,24
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)

        union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(union)

        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)
        t_conv_regression = self.t_conv_r2(t_detection_feature)

        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)  # [1, 256, 22, 22]
        rgb_pred_score = F.conv2d(rgb_conv_scores, self.rgb_score_filters, groups=N).reshape(N, 10,
                                                                                             self.score_displacement + 1,
                                                                                             self.score_displacement + 1)  # [1, 10, 19, 19], self.score_filters.shape ==[10,256, 4, 4]
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4,
                                             self.score_displacement + 4) # [1, 256, 22, 22]
        t_pred_score = F.conv2d(t_conv_scores, self.t_score_filters, groups=N).reshape(N, 10,
                                                                                       self.score_displacement + 1,
                                                                                       self.score_displacement + 1)
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)  # [1, 256, 22, 22]
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)  # [1, 256, 22, 22]
        rgb_pred_regression = self.rgb_regress_adjust(
            F.conv2d(rgb_conv_reg, self.rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))  # [1, 20, 19, 19]
        t_pred_regression = self.t_regress_adjust(
            F.conv2d(t_conv_reg, self.t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))  # [1, 20, 19, 19]

        attn_rgb_conv_score = self.attn_rgb_conv_cls2(
            attn_rgb_detection_feature)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(
            attn_rgb_detection_feature)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)

        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, self.attn_rgb_score_filters, groups=N).reshape(N, 10,
                                                                                             self.score_displacement + 1,
                                                                                             self.score_displacement + 1)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4,
                                             self.score_displacement + 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, self.attn_t_score_filters, groups=N).reshape(N, 10,
                                                                                       self.score_displacement + 1,
                                                                                       self.score_displacement + 1)
        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                               self.score_displacement + 4)
        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(
            F.conv2d(attn_rgb_conv_reg, self.attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))
        attn_t_pred_regression = self.attn_t_regress_adjust(
            F.conv2d(attn_t_conv_reg, self.attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                       self.score_displacement + 1))

        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, \
               rgb_pred_regression + t_pred_regression
