import os
import numpy as np

root_dir = '/dataset/GTOT/'
list = 'gtot.txt'
list_file = os.path.join(root_dir, list)
target_dir = '/dataset/LasHeR3/'

with open(list_file, 'r') as f:
    seq_names = f.read().strip().split('\n')
"""
init_path = os.path.join(root_dir, seq_names[0], 'groundTruth_i.txt')
temp = np.loadtxt(init_path, delimiter=' ')
for gt in temp:
    gt[2] = gt[2] - gt[0]
    gt[3] = gt[3] - gt[1]
print(temp)
np.savetxt('/home/uncleyoung/temp.txt', temp, fmt='%d',delimiter=',')
"""
for name in seq_names:
    target_path = os.path.join(target_dir, name)

    init_path = os.path.join(root_dir, name, 'init.txt')
    infrared_path = os.path.join(root_dir, name, 'groundTruth_i.txt')
    visible_path = os.path.join(root_dir, name, 'groundTruth_v.txt')

    init_gt = np.loadtxt(init_path, delimiter='\t')
    infrared_gt = np.loadtxt(infrared_path, delimiter=' ')
    for gt in infrared_gt:
        gt[2] = gt[2] - gt[0]
        gt[3] = gt[3] - gt[1]
    visible_gt = np.loadtxt(visible_path, delimiter=' ')
    for gt in visible_gt:
        gt[2] = gt[2] - gt[0]
        gt[3] = gt[3] - gt[1]
    np.savetxt(target_path + '/' + 'init.txt', init_gt, fmt='%d',delimiter=',')
    np.savetxt(target_path + '/' + 'infrared.txt', infrared_gt, fmt='%d', delimiter=',')
    np.savetxt(target_path + '/' + 'visible.txt', visible_gt, fmt='%d', delimiter=',')

"""
    cur_path = os.path.join(root_dir, name)
    file_list = os.listdir(cur_path)
    for i in file_list:
        if i == 'init.txt':
            os.rename(cur_path + '/' + i, cur_path + '/' + 'visible')
        if i == 'groundTruth_i.txt':
            os.rename(cur_path + '/' + i, cur_path + '/' + 'infrared.txt')
        if i == 'groundTruth_v.txt':
            os.rename(cur_path + '/' + i, cur_path + '/' + 'visible.txt')
"""