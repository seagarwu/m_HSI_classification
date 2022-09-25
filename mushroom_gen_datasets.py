# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:01:54 2020

@author: User
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = './Mask_RCNN/mushroom/'
sys.path.append(ROOT_DIR)  # To find local version of the library
os.chdir(ROOT_DIR)

from mushroom_globalvars import _init, set_value
import mushroom_mrcnn


BIG_IMAGE_FILENAME = 'D:/吳鼎榮/林菇/NIR/採收菇1/MAT/stage1_air_1_NIR_RT.mat'
# DATASET_DIRNAME = 'E:/我的雲端硬碟/dataset/mushroom_2020.5.23/NIR/'
DATASET_DIRNAME = 'D:/datasets/'
PNG_DIRNAME = '../datasets/mushroom/'
REMOVED_FIRSTBAND = 9
REMOVED_LASTBAND = 453  # 433:出來的洋菇看起來不錯
SHOWN_BAND = 300  # 400: mrcnn會選不到其中一顆洋菇。這邊未來可以改進，自動選擇大圖edge容易識別的band (也必須達到dnn acc 100%才行，因為目前是100%)

#
def npy2png300(fullfilename, shown_band):
    from PIL import Image 
    dirname = os.path.dirname(fullfilename)
    basename = os.path.basename(fullfilename)
    filename, file_extension = os.path.splitext(basename)
    images = np.load(fullfilename)   
    image = images[shown_band]
    rescaled = (255.0 / np.nanmax(image) * (image - np.nanmin(image))).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im = im.convert('RGB') # 從8 bits轉成24 bits
    return im
    
    
#
def get_stage_name(filename):
    pos = filename.find('stage')
    stage_name = filename[pos:pos+6]  # e.g. stage2
    return stage_name

#
def get_filename_max_num(base_path, stage_name):
    import os
    import re
    import fnmatch
    
    regex = re.compile(r'\d+')
    filename_num = [0]  # 一定要有一個值，如果設成[]，當都沒有符合的檔案時會有error
    
    for entry in os.listdir(base_path):
        match_filename = stage_name + '*.npy'  # e.g. stage2.1.npy其中的1是必須符合stage2 
        if fnmatch.fnmatch(entry, match_filename):
            if os.path.isfile(os.path.join(base_path, entry)):
                xx = [int(x) for x in regex.findall(entry)]
                if len(xx) > 1:  # e.g. stage2.1.npy -> [2, 1], len(xx)=2
                    xx = xx[1]  # 取1 
                filename_num.append(xx)
    
    filename_num = np.array(filename_num)
    max_num = max(filename_num)  # 當filename_num=[]時做max()會出現error   
    return max_num

#
if __name__ == '__main__':
    
    ## 輸入大圖的檔名和要儲存datasets的路徑 (可用逗號輸入多個檔案)
    big_image_filenames = input('Big image filenames [' + BIG_IMAGE_FILENAME + ']: ') or BIG_IMAGE_FILENAME
    dataset_dirname = input('Datasets path [' + DATASET_DIRNAME + ']: ') or DATASET_DIRNAME
    first_band = input('First band to remove [begin-' + str(REMOVED_FIRSTBAND) + ']:') or REMOVED_FIRSTBAND
    last_band = input('Last band to remove [' + str(REMOVED_LASTBAND) + '-end]:') or REMOVED_LASTBAND
    shown_band = input('Shown band [' + str(SHOWN_BAND) + ']:') or SHOWN_BAND

    print('Starting...')
    first_band = int(first_band)    
    last_band = int(last_band)    
    shown_band = int(shown_band)    
    
    big_image_filenames = [x for x in big_image_filenames.split(',')]  # str: 'E:/test.mat' -> ['E:/test.mat',]
    folder_no = np.shape(big_image_filenames)[0]  # type(np.shape(folder_name)) is tuple, type(np.shape(folder_name)[0]) is int 
    
    # 
    for q in range(folder_no):
        big_image_filename = big_image_filenames[q]
        print('\n<<' + big_image_filename + '>>')

        ## load 洋菇大圖
        import h5py
        from skimage import filters
        
        basename = os.path.basename(big_image_filename)
        filename, _ = os.path.splitext(basename)    
        stage_name = get_stage_name(big_image_filename)
        
        images=h5py.File(big_image_filename)
        images=np.array(images['data'])
        images=np.transpose(images,(0,2,1))  # (512, 1636, 640) 含noise band和背景雜訊
        
        
        ## de-noise bands
        images = images[:last_band,] # 將 NIR 圖去掉雜訊的band，1-9, 454-512 band (還剩下背景雜訊)
        images = images[first_band:,] # 剩 444 bands -> (444, 1636, 640)
        plt.imshow(images[shown_band]) # show洋菇大圖 (含背景雜訊)
        plt.show()
        print('洋菇大圖 (已去除band noise，但仍含background noise)')
        
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
        plt.imshow(images[shown_band]) # show洋菇大圖 (已不含背景雜訊)
        plt.show()    
        print('洋菇大圖 (已去除band noise和background noise)')
        
        
        ## 洋菇大圖轉出band 300到png檔
        from PIL import Image
        im = rescaled * mask  # 讓圖也先濾過雜訊再給mask-rcnn辨識，這樣準確度會更高，train mask-rcnn的epochs也可以縮短
        im = im.reshape((band, height, width), order='F') 
        im = Image.fromarray(im[shown_band])  # 只要band 300或其它band轉成png檔即可
        im = im.convert('RGB') # 從8 bits轉成24 bits
        png_filename = PNG_DIRNAME + filename + '_' + str(shown_band) + '.png'
        im.save(png_filename)
    
    
        ## 跑以下程式前要確認已經跑過mask-rcnn訓練程式 (mushroom_mrcnn_train.py) 產生weighting file (mask_rcnn_mushroom_xxxx.h5)
        ## mushroom_mrcnn_train.py的目的是為了準確抓到每顆菇的位置，和不要抓到洞的位置
        ## load洋菇大圖png image跑mask-rcnn程式 (detection)，將mask及roi儲存到detect_result.pkl
        _init()
        set_value('args', argparse.Namespace(command = 'splash', weights = 'last', 
                                             image = png_filename,  
                                             dataset = 'None', logs = '..\logs', video = 'None')
                 )        
        mushroom_mrcnn.cmd_run() # show洋菇大圖含mask & bounding box
        print('(Mask-RCNN框選到的洋菇)')
        
    
        ## 開啟detect_result.pkl，根據mask和roi取出洋菇大圖中的所有洋菇，儲存在datasets path中
        import pickle
        
        with open('detect_result.pkl', 'rb') as f: #  檔案內容來自balloon.py中r = model.detect([image], verbose=1)[0]的結果
            r = pickle.load(f)
        r['masks'].shape # (1636, 640, 21) -> 21是大圖中有21顆洋菇
    
        max_num = get_filename_max_num(dataset_dirname, stage_name)  # e.g. 如果stage2最後的檔案是stage2.10.npy，則max_num=10，下次有新的dataset時則從stage2.11.npy開始
        mush_num = r['masks'].shape[2]  # 大圖的洋菇顆數
        for i in range(mush_num):  # 將大圖中所有detect到的洋菇都儲存成.npy
            mask=r['masks'][...,i]  # (1636, 640)  # 取出第i顆洋菇的mask
            mask = np.array(mask, int) # 將variable內容為True/False轉成int
            mask.shape  # (1636, 640)
            
            r['rois'].shape # (21, 4)
            roi=r['rois'][i]  # (4,) # 取出第i顆的座標
            
            mush = images[:,] * mask # 抓出所要的洋菇
            mush = mush[:, roi[0]:roi[2], roi[1]:roi[3]] # 利用roi抓出大圖中洋菇的位置 -> (444, 1636, 640)
            plt.figure()
            plt.imshow(mush[shown_band])  # show每一個顆洋菇
            plt.show()
    
            dataset_filename = stage_name + '.' + np.str(i + max_num + 1) + '.npy'  # 從1開始算
            print(dataset_filename)
            np.save(dataset_dirname + dataset_filename, mush)
            
            ## 將.npy轉成.png (band 300)，用來檢查影像是不是洋菇
            png_image = npy2png300(dataset_dirname + dataset_filename, shown_band)
            filename, _ = os.path.splitext(dataset_filename)
            png_filename = filename + '.png'
            png_image.save(dataset_dirname + png_filename)
        
    
    print('\nFinished.')    





