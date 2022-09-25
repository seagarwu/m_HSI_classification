# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:18:00 2020

@author: Alan Wu
"""

import h5py
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt  
import os  

BIG_IMAGE_FILENAME = 'D:/吳鼎榮/林菇/NIR/採收菇1/MAT/stage1_air_1_NIR_RT.mat'
PNG_DIRNAME = './Mask_RCNN/mushroom/tmp/'
SAVED_BAND = 300

if __name__ == '__main__':
    big_image_filename = input('Big image filename [' + BIG_IMAGE_FILENAME + ']: ') or BIG_IMAGE_FILENAME
    saved_band = input('Band [' + str(SAVED_BAND) + ']: ') or SAVED_BAND

    print('Starting...')    

    ## load 洋菇大圖
    images=h5py.File(big_image_filename)
    images=np.array(images['data'])
    images=np.transpose(images,(0,2,1))  # (512, 1636, 640) 含noise band和背景雜訊
    
    ## de-background noise of all bands using Otsu
    band = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    images = images.reshape((band, height*width), order='F')  # (444, 1047040)
    rescaled = (255.0 / np.nanmax(images) * (images - np.nanmin(images))).astype(np.uint8)  # 目的是把nan去掉才能做otsu
    val = filters.threshold_otsu(rescaled)
    mask = rescaled > val  # (444, 1047040)
    
    images = images * mask  # 去掉背景雜訊
    images = images.reshape((band, height, width), order='F')
    plt.imshow(images[saved_band]) # show洋菇大圖 (已不含背景雜訊)
    plt.show()    
    
    
    ## 洋菇大圖轉出band 300到png檔
    from PIL import Image
    im = rescaled * mask  # 讓圖也先濾過雜訊再給mask-rcnn辨識，這樣準確度會更高，train mask-rcnn的epochs也可以縮短
    im = im.reshape((band, height, width), order='F') 
    im = Image.fromarray(im[saved_band])  # 只要band 300或其它band轉成png檔即可
    im = im.convert('RGB') # 從8 bits轉成24 bits
    basename = os.path.basename(big_image_filename)
    filename, _ = os.path.splitext(basename)    
    png_filename = PNG_DIRNAME + filename + '_' + str(saved_band) + '.png'
    im.save(png_filename)
    
    print('\nFinished.')