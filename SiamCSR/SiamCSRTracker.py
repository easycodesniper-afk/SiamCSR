import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from tqdm import tqdm
from .rgbt_network import SiamCSRNet
from .config import config
from .transforms import ToTensor
from .utils import generate_anchors, get_exemplar_image, get_instance_image, box_transform_inv,add_box_img,add_box_img_left_top,show_image

from IPython import embed

torch.set_num_threads(1)  # otherwise pytorch will take all cpus


def change(r):
    return np.maximum(r, 1. / r)

def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)

def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)

def show_rgbt_image(img, boxes=None, gt=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
        if gt is not None:
            gt = np.array(gt, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
    if gt is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        gt = np.array(gt, dtype=np.int32)
        if gt.ndim == 1:
            gt = np.expand_dims(gt, axis=0)
        if box_fmt == 'ltrb':
            gt[:, 2:] -= gt[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        gt[:, :2] = np.clip(gt[:, :2], 0, bound)
        gt[:, 2:] = np.clip(gt[:, 2:], 0, bound - gt[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            gt1 = (gt[i][0], gt[i][1])
            gt2 = (gt[i][0] + gt[i][2], gt[i][1] + gt[i][3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
            img = cv2.rectangle(img, gt1, gt2, (255, 215, 0), 2)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


class SiamCSRTracker:
    def __init__(self, model_path, cfg=None, is_deterministic=False):
        self.name = 'SiamCSR'
        if cfg:
            config.update(cfg)
        self.is_deterministic = is_deterministic
        model = SiamCSRNet()
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model = model.cuda()
        self.model = model
        self.model.eval()

        self.rgb_transforms = transforms.Compose([ToTensor()])
        self.t_transforms = transforms.Compose([ToTensor()])

        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        config.valid_scope)  #
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, rgb_frame, t_frame, bbox, bbox_r, bbox_t):
        self.bbox = np.array([bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2, bbox[2], bbox[3]])
        br = np.array([bbox_r[0] - 1 + (bbox_r[2] - 1) / 2, bbox_r[1] - 1 + (bbox_r[3] - 1) / 2, bbox_r[2], bbox_r[3]])
        bt = np.array([bbox_t[0] - 1 + (bbox_t[2] - 1) / 2, bbox_t[1] - 1 + (bbox_t[3] - 1) / 2, bbox_t[2], bbox_t[3]])

        self.pos = np.array([bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])# center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])# width, height
        self.origin_target_sz = np.array([bbox[2], bbox[3]])  # w,h
        self.rgb_img_mean = np.mean(rgb_frame, axis=(0, 1))
        rgb_exemplar_img, _, _ = get_exemplar_image(rgb_frame, br, config.exemplar_size, config.context_amount,
                                                      self.rgb_img_mean)
        self.t_img_mean = np.mean(t_frame, axis=(0, 1))
        t_exemplar_img, _, _ = get_exemplar_image(t_frame, bt, config.exemplar_size, config.context_amount,
                                                  self.t_img_mean)
        rgb_exemplar_img = self.rgb_transforms(rgb_exemplar_img)[None, :, :, :]
        t_exemplar_img = self.t_transforms(t_exemplar_img)[None, :, :, :]
        self.model.track_init(rgb_exemplar_img.cuda(), t_exemplar_img.cuda())

    def update(self, rgb_frame, t_frame):
        rgb_instance_img, _, _, scale_x = get_instance_image(rgb_frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.rgb_img_mean)
        t_instance_img, _, _, _ = get_instance_image(t_frame, self.bbox, config.exemplar_size,
                                                        config.instance_size,
                                                        config.context_amount, self.t_img_mean)
        rgb_instance_img = self.rgb_transforms(rgb_instance_img)[None, :, :, :]
        t_instance_img = self.t_transforms(t_instance_img)[None, :, :, :]
        pred_score_1, pred_score_2, pred_regression = self.model.track(rgb_instance_img.cuda(), t_instance_img.cuda())

        pred_conf_1 = pred_score_1.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0, 2,1)
        pred_offset = pred_regression.reshape(-1, 4, config.anchor_num * config.score_size * config.score_size).permute(0, 2, 1)
        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred_1 = F.softmax(pred_conf_1, dim=2)[0, :, 1].cpu().detach().numpy()

        pred_conf_2 = pred_score_2.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0, 2, 1)

        score_pred_2 = F.softmax(pred_conf_2, dim=2)[0, :, 1].cpu().detach().numpy()

        s_c_1 = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))
        r_c_1 = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))
        penalty_1 = np.exp(-(r_c_1 * s_c_1 - 1.) * config.penalty_k)
        pscore_1 = penalty_1 * score_pred_1
        pscore_1 = pscore_1 * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id_1 = np.argmax(pscore_1)

        s_c_2 = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))
        r_c_2 = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))
        penalty_2 = np.exp(-(r_c_2 * s_c_2 - 1.) * config.penalty_k)
        pscore_2 = penalty_2 * score_pred_2
        pscore_2 = pscore_2 * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id_2 = np.argmax(pscore_2)

        if pscore_1[best_pscore_id_1] >= pscore_2[best_pscore_id_2]:
            best_pscore_id = best_pscore_id_1
            penalty = penalty_1
            score_pred = score_pred_1

        else:
            best_pscore_id = best_pscore_id_2
            penalty = penalty_2
            score_pred = score_pred_2

        target = box_pred[best_pscore_id, :] / scale_x
        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, rgb_frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, rgb_frame.shape[0])
        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])

        self.bbox = (  # cx, cy, w, h
            np.clip(bbox[0], 0, rgb_frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, rgb_frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, rgb_frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, rgb_frame.shape[0]).astype(np.float64))

        bbox = np.array([  # tr-x,tr-y w,h
            self.pos[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.pos[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.target_sz[0], self.target_sz[1]])

        return bbox

    def track(self, rgb_img_files, t_img_files, groundtruth, bbox_rgb, bbox_t, dataset, visualize=False):
        frame_num = len(rgb_img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = groundtruth[0]

        # GTOT
        if dataset == 'GTOT':
            bbox_t = (bbox_t[0], bbox_t[1], bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1])
            bbox_rgb = (bbox_rgb[0], bbox_rgb[1], bbox_rgb[2] - bbox_rgb[0], bbox_rgb[3] - bbox_rgb[1])
        times = np.zeros(frame_num)

        for f, rgb_img_file in enumerate(rgb_img_files):
            t_img_file = t_img_files[f]
            rgb_img = cv2.imread(rgb_img_file, cv2.IMREAD_COLOR)
            t_img = cv2.imread(t_img_file, cv2.IMREAD_GRAYSCALE)
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
            begin = time.time()
            if f == 0:
                self.init(rgb_img, t_img, boxes[0], bbox_rgb, bbox_t)
            else:
                boxes[f, :] = self.update(rgb_img, t_img)
            times[f] = time.time() - begin
            if visualize:
                show_rgbt_image(rgb_img, boxes[f, :], gt=groundtruth[f], fig_n=1, cvt_code=cv2.COLOR_RGB2BGR)
                show_rgbt_image(t_img, boxes[f, :], gt=groundtruth[f], fig_n=2, cvt_code=cv2.COLOR_RGB2GRAY)
        return boxes, times