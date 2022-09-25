# -*- coding: utf-8 -*-
"""
 準確抓到每顆菇的位置，和不要抓到洞的位置

@author: Alan Wu
"""

import argparse
from mushroom_globalvars import _init, set_value
import mushroom_mrcnn

DATASET_DIRNAME = '../datasets/mushroom'
WEIGHT_FILENAME = 'last'

if __name__ == '__main__':
    
    _init()
    
    dataset_dirname = input('Datasets path [' + DATASET_DIRNAME + ']: ') or DATASET_DIRNAME
    weight_filename = input('Weight filename [' + WEIGHT_FILENAME + ']: ') or WEIGHT_FILENAME
    
    # set_value('args', argparse.Namespace(command = 'train', weights = '../mask_rcnn_balloon.h5',
    set_value('args', argparse.Namespace(command = 'train', weights = weight_filename,
                                         dataset = dataset_dirname, image = 'None',
                                         logs = '..\logs', video = 'None')  
             )                           # 不能用../log 不然在儲存../log/xxx/train/plugins/profile/xxx時會有問題
        
    mushroom_mrcnn.cmd_run()
    
    print('Finished.')