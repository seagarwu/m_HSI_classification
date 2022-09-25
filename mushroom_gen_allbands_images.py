# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:50:18 2020

@author: Alan Wu
"""

import mushroom_dnn as mush

BIG_IMAGE_FILENAME = 'D:/吳鼎榮/林菇/NIR/採收菇1/MAT/stage1_air_1_NIR_RT.mat'

if __name__ == '__main__':
    big_image_filename = input('Big Image filename [' + BIG_IMAGE_FILENAME + ']: ') or BIG_IMAGE_FILENAME

    print('Starting...')
    mush.gen_pngs_from_npy(big_image_filename)
    

print('\nFinished.')