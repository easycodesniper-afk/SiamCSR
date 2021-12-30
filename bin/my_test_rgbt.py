from __future__ import absolute_import

import os
import sys
#sys.path.append('/XXX/SiamCSR')
from SiamCSR.SiamCSRTracker import SiamCSRTracker
from data.rgbt234 import RGBT234
from data.gtot import GTOT
import argparse
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')

    # print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='rgbt tracking')


    parser.add_argument('--model_path', default='../models/snapshot_gtot.pth', type=str, help='eval one special video')
    parser.add_argument('--result_dir', default='../result/GTOT', type=str)
    parser.add_argument('--testing_dataset', default='GTOT', type=str)
    args = parser.parse_args()

    tracker = SiamCSRTracker(args.model_path)

    root_dir = '/dataset/GTOT'
    rgbt_sequence = GTOT(root_dir)
    print('Running tracker %s on %s...' % (tracker.name, type(rgbt_sequence).__name__))
    for s, (rgb_img_files, t_img_files, groundtruth, gt_r, gt_t) in tqdm(enumerate(rgbt_sequence), total=len(rgbt_sequence)):
        seq_name = rgbt_sequence.seq_names[s]
        record_file = os.path.join(args.result_dir, tracker.name, '%s.txt' % seq_name)
        boxes, times = tracker.track(rgb_img_files, t_img_files, groundtruth, gt_r[0], gt_t[0], args.testing_dataset, visualize=True)
        toc = sum(times)
        print('(Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            seq_name, toc, len(rgb_img_files) / toc))
        _record(record_file, boxes, times)