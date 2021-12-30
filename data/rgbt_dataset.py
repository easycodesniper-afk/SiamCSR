# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import os.path as osp
from SiamCSR.utils import *
from SiamCSR.config import config
from torch.utils.data import Dataset
from PIL import Image

class RGBTDataset(Dataset):
    def __init__(self, seq_dataset, z_r_transforms, z_t_transforms,\
                 x_r_transforms, x_t_transforms, name = 'RGB-T234'):
        self.max_inter     = 100 #100, like T in SiamFC,The images are extracted from two frames of a video that both contain the object and are at most T frames apart
        self.z_r_transforms  = z_r_transforms
        self.z_t_transforms = z_t_transforms
        self.x_r_transforms  = x_r_transforms
        self.x_t_transforms = x_t_transforms
        self.sub_class_dir = seq_dataset
        self.ret           = {}
        self.count         = 0
        self.length = len(seq_dataset)
        self.name          = name
        self.anchors       = generate_anchors( config.total_stride,
                                                    config.anchor_base_size,
                                                    config.anchor_scales,
                                                    config.anchor_ratios,
                                                    config.score_size)

    def _pick_rgb_and_t_pairs(self, index_of_subclass):
        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'
        video_name_rgb = self.sub_class_dir[index_of_subclass][0]
        video_name_t = self.sub_class_dir[index_of_subclass][1]
        video_num = len(video_name_rgb)
        video_gt_rgb = self.sub_class_dir[index_of_subclass][2]
        video_gt_t = self.sub_class_dir[index_of_subclass][3]
        status = True
        while status:
            if self.max_inter >= video_num - 1:
                self.max_inter = video_num // 2
            template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num - 1) # limit template_index from 0 to video_num - 1
            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num - 1)# limit detection_index from 0 to video_num - 1

            template_path_rgb, detection_path_rgb  = video_name_rgb[template_index], video_name_rgb[detection_index]
            template_path_t, detection_path_t = video_name_t[template_index], video_name_t[detection_index]

            template_gt_rgb  = video_gt_rgb[template_index]
            detection_gt_rgb = video_gt_rgb[detection_index]
            template_gt_t = video_gt_t[template_index]
            detection_gt_t = video_gt_t[detection_index]

            if template_gt_rgb[2] * template_gt_rgb[3] * detection_gt_rgb[2] * detection_gt_rgb[3] \
                    * template_gt_t[2] * template_gt_t[3] * detection_gt_t[2] * detection_gt_t[3] != 0:
                status = False

        # load infomation of template and detection
        self.ret['template_path_rgb'] = template_path_rgb
        self.ret['detection_path_rgb'] = detection_path_rgb

        self.ret['template_path_t'] = template_path_t
        self.ret['detection_path_t'] = detection_path_t

        self.ret['template_target_x1y1wh_rgb'] = template_gt_rgb
        self.ret['detection_target_x1y1wh_rgb']= detection_gt_rgb
        self.ret['template_target_x1y1wh_t'] = template_gt_t
        self.ret['detection_target_x1y1wh_t'] = detection_gt_t

        t1, t2 = self.ret['template_target_x1y1wh_rgb'].copy(), self.ret['detection_target_x1y1wh_rgb'].copy()
        self.ret['template_target_xywh_rgb'] = np.array([t1[0] + t1[2] // 2, t1[1] + t1[3] // 2, t1[2], t1[3]], np.float32) # (cx, cy, w, h)
        self.ret['detection_target_xywh_rgb'] = np.array([t2[0] + t2[2] // 2, t2[1] + t2[3] // 2, t2[2], t2[3]], np.float32)
        t1, t2 = self.ret['template_target_x1y1wh_t'].copy(), self.ret['detection_target_x1y1wh_t'].copy()
        self.ret['template_target_xywh_t'] = np.array([t1[0] + t1[2] // 2, t1[1] + t1[3] // 2, t1[2], t1[3]],
                                                        np.float32)  # (cx, cy, w, h)
        self.ret['detection_target_xywh_t'] = np.array([t2[0] + t2[2] // 2, t2[1] + t2[3] // 2, t2[2], t2[3]],
                                                         np.float32)
        self.ret['anchors'] = self.anchors

    def open(self):
        '''template'''
        template_img_rgb = cv2.imread(self.ret['template_path_rgb'], cv2.IMREAD_COLOR)
        template_img_t = cv2.imread(self.ret['template_path_t'], cv2.IMREAD_COLOR)
        detection_img_rgb = cv2.imread(self.ret['detection_path_rgb'], cv2.IMREAD_COLOR)
        detection_img_t = cv2.imread(self.ret['detection_path_t'], cv2.IMREAD_COLOR)

        if np.random.rand(1) < config.gray_ratio: # why?
            template_img_rgb = cv2.cvtColor(template_img_rgb, cv2.COLOR_RGB2GRAY)
            template_img_rgb = cv2.cvtColor(template_img_rgb, cv2.COLOR_GRAY2RGB)
            detection_img_rgb = cv2.cvtColor(detection_img_rgb, cv2.COLOR_RGB2GRAY)
            detection_img_rgb = cv2.cvtColor(detection_img_rgb, cv2.COLOR_GRAY2RGB)

        img_mean_rgb_tem = np.mean(template_img_rgb, axis=(0, 1))
        img_mean_t_tem = np.mean(template_img_t, axis=(0, 1))

        exemplar_img_r, scale_z, s_z, w_x, h_x = self.get_exemplar_image(template_img_rgb,
                                                                        self.ret['template_target_xywh_rgb'],
                                                                        config.exemplar_size,
                                                                        config.context_amount,
                                                                        img_mean_rgb_tem)
        exemplar_img_t, _, _, _, _ = self.get_exemplar_image(template_img_t,
                                                             self.ret['template_target_xywh_t'],
                                                             config.exemplar_size,
                                                             config.context_amount,
                                                             img_mean_t_tem)
        self.ret['exemplar_img_r'], self.ret['exemplar_img_t'] = \
                        exemplar_img_r, exemplar_img_t

        '''detection'''
        detection_bb_rgb = self.ret['detection_target_xywh_rgb']
        cx, cy, w, h = detection_bb_rgb  # float type

        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        s_x = s_z / (config.instance_size // 2)

        img_mean_rgb_det = np.mean(detection_img_rgb, axis=(0, 1))
        img_mean_t_det = np.mean(detection_img_t, axis=(0, 1))

        a_x_ = np.random.choice(range(-12,12))
        a_x = a_x_ * s_x

        b_y_ = np.random.choice(range(-12,12))
        b_y = b_y_ * s_x

        instance_img_r, _, _, w_x, h_x, _ = self.get_instance_image(
                                                                    detection_img_rgb,
                                                                    detection_bb_rgb,
                                                                    config.exemplar_size, # 127
                                                                    config.instance_size,# 255
                                                                    config.context_amount,           # 0.5
                                                                    a_x, b_y,
                                                                    img_mean_rgb_det)

        size_x = config.instance_size
        x1, y1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
        x2, y2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2

        if x1 > x2 :
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        w_rgb = x2 - x1
        h_rgb = y2 - y1

        self.ret['instance_img_r'] = instance_img_r


        detection_bb_t = self.ret['detection_target_xywh_t']
        cx, cy, w, h = detection_bb_t  # float type

        wc_z_t = w + 0.5 * (w + h)
        hc_z_t = h + 0.5 * (w + h)
        s_z_t = np.sqrt(wc_z_t * hc_z_t)
        s_x_t = s_z_t / (config.instance_size // 2)
        a_x_t = a_x_ * s_x_t
        b_y_t = b_y_ * s_x_t

        instance_img_t, _, _, w_x_t, h_x_t, _ = self.get_instance_image(
                                                    detection_img_t,
                                                    detection_bb_t,
                                                    config.exemplar_size,  # 127
                                                    config.instance_size,  # 255
                                                    config.context_amount,  # 0.5
                                                    a_x_t, b_y_t,
                                                    img_mean_t_det)

        size_x_t = config.instance_size
        x1_t, y1_t = (size_x_t + 1) / 2 - w_x_t / 2, (size_x_t + 1) / 2 - h_x_t / 2
        x2_t, y2_t = (size_x_t + 1) / 2 + w_x_t / 2, (size_x_t + 1) / 2 + h_x_t / 2

        if x1_t > x2_t:
            x1_t, x2_t = x2_t, x1_t
        if y1_t > y2_t:
            y1_t, y2_t = y2_t, y1_t

        w_t = x2_t - x1_t
        h_t = y2_t - y1_t

        self.ret['cx, cy, w, h'] = [int(a_x_), int(b_y_), int((w_rgb + w_t) / 2), int((h_rgb + h_t) / 2)]
        self.ret['instance_img_t'] = instance_img_t


    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        cx, cy, w, h = bbox

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = size_z / s_z

        exemplar_img, scale_x = self.crop_and_pad_old(img, cx, cy, size_z, s_z, img_mean) # mapping origin width\heigth to w_x\h_x

        w_x = w * scale_x
        h_x = h * scale_x

        return exemplar_img, scale_z, s_z, w_x, h_x

    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, a_x, b_y, img_mean=None):

        cx, cy, w, h = bbox  # float type

        #cx, cy = cx - a_x , cy - b_y
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z) # the width of the crop box

        scale_z = size_z / s_z

        s_x = s_z * size_x / size_z
        instance_img, gt_w, gt_h, scale_x, scale_h, scale_w = self.crop_and_pad(img, cx, cy, w, h, a_x, b_y,  size_x, s_x, img_mean)
        w_x = gt_w #* scale_x #w * scale_x
        h_x = gt_h #* scale_x #h * scale_x

        a_x, b_y = a_x*scale_w, b_y*scale_h

        return instance_img, a_x, b_y, w_x, h_x, scale_x

    def crop_and_pad(self, img, cx, cy, gt_w, gt_h, a_x, b_y, model_sz, original_sz, img_mean=None):

        #random = np.random.uniform(-0.15, 0.15)
        scale_h = 1.0 + np.random.uniform(- 0.15, 0.15)
        scale_w = 1.0 + np.random.uniform(- 0.15, 0.15)

        im_h, im_w, _ = img.shape

        xmin = (cx - a_x) - ((original_sz - 1) / 2) * scale_w
        xmax = (cx - a_x) + ((original_sz - 1) / 2) * scale_w

        ymin = (cy - b_y) - ((original_sz - 1) / 2)* scale_h
        ymax = (cy - b_y) + ((original_sz - 1) / 2)* scale_h

        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))

        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization

            te_im[:, :, :] = img_mean
            te_im[top:top + r, left:left + c, :] = img

            if top:
                te_im[0 : top, left : left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left : left + c, :] = img_mean
            if left:
                te_im[:, 0 : left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean

            im_patch_original = te_im[int(ymin) : int(ymax + 1), int(xmin) : int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin) : int((ymax) + 1), int(xmin) : int((xmax) + 1), :]

        if not np.array_equal(model_sz, original_sz):
            h, w, _ = im_patch_original.shape
            if h < w:
                scale_h_ = 1
                scale_w_ = h / w
                scale = config.instance_size / h
            elif h > w:
                scale_h_ = w / h
                scale_w_ = 1
                scale = config.instance_size / w
            elif h == w:
                scale_h_ = 1
                scale_w_ = 1
                scale = config.instance_size / w

            gt_w = gt_w * scale_w_
            gt_h = gt_h * scale_h_

            gt_w = gt_w * scale
            gt_h = gt_h * scale

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        return im_patch, gt_w, gt_h, scale, scale_h_, scale_w_

    def crop_and_pad_old(self, img, cx, cy, model_sz, original_sz, img_mean=None):
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2
        ymax = ymin + original_sz - 1

        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))
        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]
        return im_patch, scale

    def round_up(self, value):
        return round(value + 1e-6 + 1000) - 1000

    def _target(self):
        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, self.ret['cx, cy, w, h']))))
        return regression_target, conf_target

    def compute_target(self, anchors, box):
        regression_target = self.box_transform(anchors, box)
        iou = self.compute_iou(anchors, box).flatten()

        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]

        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0

        return regression_target, label

    def box_transform(self, anchors, gt_box):
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box

        target_x = (gt_cx - anchor_xctr) / anchor_w
        target_y = (gt_cy - anchor_yctr) / anchor_h
        target_w = np.log(gt_w / anchor_w)
        target_h = np.log(gt_h / anchor_h)
        regression_target = np.hstack((target_x, target_y, target_w, target_h))
        return regression_target

    def compute_iou(self, anchors, box):
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                               axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return iou

    def _tranform(self):
        self.ret['exemplar_img_r'] = self.z_r_transforms(self.ret['exemplar_img_r'])
        self.ret['exemplar_img_t'] = self.z_t_transforms(self.ret['exemplar_img_t'])
        self.ret['instance_img_r'] = self.x_r_transforms(self.ret['instance_img_r'])
        self.ret['instance_img_t'] = self.x_t_transforms(self.ret['instance_img_t'])

    def __getitem__(self, index):
        index = random.choice(range(len(self.sub_class_dir)))
        self._pick_rgb_and_t_pairs(index)
        self.open()
        self._tranform()
        regression_target, conf_target = self._target()
        """
        self.count += 1
        if self.count >= self.length:
            self.count = 0
        """
        return self.ret['exemplar_img_r'], self.ret['exemplar_img_t'], self.ret['instance_img_r'],\
               self.ret['instance_img_t'], regression_target, conf_target.astype(np.int64)

    def __len__(self):
        return config.pairs_num_per_epoch