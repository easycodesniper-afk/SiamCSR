import os
import sys

from SiamCSR.SiamCSRTracker import SiamCSRTracker

import numpy as np
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _record(record_file, boxes, times):

    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')

    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')

if __name__=='__main__':
    tracker = SiamCSRTracker('../models/snapshot_gtot.pth')

    demo_seq_dir = 'Quarreling'
    seq_dirs_rgb = os.path.join(demo_seq_dir, 'v')
    seq_dirs_t = os.path.join(demo_seq_dir, 'i')
    anno_files = os.path.join(demo_seq_dir, 'init.txt')
    anno_files_rgb = os.path.join(demo_seq_dir, 'groundTruth_v.txt')
    anno_files_t = os.path.join(demo_seq_dir, 'groundTruth_i.txt')

    img_files_rgb = sorted(glob.glob(os.path.join(
        seq_dirs_rgb, '*.*')))
    img_files_t = sorted(glob.glob(os.path.join(
        seq_dirs_t, '*.*')))
    anno = np.loadtxt(anno_files, delimiter='\t')
    anno_rgb = np.loadtxt(anno_files_rgb, delimiter=' ')
    anno_t = np.loadtxt(anno_files_t, delimiter=' ')

    seq_name = 'Quarreling'
    record_file = os.path.join(tracker.name, '%s.txt' % seq_name)
    boxes, times = tracker.track(img_files_rgb, img_files_t, anno, anno_rgb[0], anno_t[0], 'GTOT',
                                 visualize=True)
    toc = sum(times)
    print('(Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        seq_name, toc, len(img_files_rgb) / toc))
    _record(record_file, boxes, times)