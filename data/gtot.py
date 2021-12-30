from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six

class GTOT(object):
    r"""
    Publication:
        RGB-t234:RGB-T Object Tracking:Benchmark and Baseline
    Args:
        root_dir:absolute path
        ...
    """

    def __init__(self, root_dir, list = 'gtot.txt'):
        super(GTOT, self).__init__()
        assert isinstance(root_dir, six.string_types)
        assert isinstance(list, six.string_types)
        self.root_dir = root_dir
        self._check_integrity(root_dir, list)
        list_file = os.path.join(root_dir, list)
        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs_rgb = [os.path.join(root_dir, s, 'v')
                             for s in self.seq_names]
        self.seq_dirs_t = [os.path.join(root_dir, s, 'i')
                           for s in self.seq_names]
        self.anno_files = [os.path.join(root_dir, s, 'init.txt')
                           for s in self.seq_names]
        self.anno_files_rgb = [os.path.join(root_dir, s, 'groundTruth_v.txt')
                           for s in self.seq_names]
        self.anno_files_t = [os.path.join(root_dir, s, 'groundTruth_i.txt')
                           for s in self.seq_names]

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index =self.seq_names.index(index)

        img_files_rgb = sorted(glob.glob(os.path.join(
            self.seq_dirs_rgb[index], '*.*')))
        img_files_t = sorted(glob.glob(os.path.join(
            self.seq_dirs_t[index], '*.*')))
        anno = np.loadtxt(self.anno_files[index], delimiter='\t')
        anno_rgb = np.loadtxt(self.anno_files_rgb[index], delimiter=' ')
        anno_t = np.loadtxt(self.anno_files_t[index], delimiter=' ')
        assert len(img_files_rgb) == len(img_files_t) and len(img_files_t) == len(anno) \
               and len(anno) == len(anno_rgb) and len(anno_rgb) == len(anno_t)

        return img_files_rgb, img_files_t, anno, anno_rgb, anno_t

    def _check_integrity(self, root_dir, list = 'GTOT.txt'):
        list_file = os.path.join(root_dir, list)
        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            raise Exception('Dataset not found or corrupted.')

if __name__ == "__main__":
    root_dir = '/dataset/GTOT'
    dataset = GTOT(root_dir = root_dir, list = 'gtot.txt')
    dataset.__getitem__(49)