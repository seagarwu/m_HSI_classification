# -*- coding: utf-8 -*-
"""
@author: Alan Wu
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from skimage import filters
# from google.colab import drive
# drive.mount('/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /gdrive/My Drive/dataset/mushroom_2020.4.23/NIR/train/stage1
# %ls

train_images = []
train_labels = []
test_images = []
test_labels = []
class_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
model=[]
w=[]
predictions=[]
show_test_images = []
show_test_basename = []

ROI_HEIGHT = 150+1  # 取NIR最大菇的size
ROI_WIDTH = 143+1
# ROI_HEIGHT = 202+1  # 取VNIR最大菇的size
# ROI_WIDTH = 202+1

# EPOCHS = 50000
EPOCHS = 20000

# index=[41,3,4,35,13,51,34,54,43,27,60,15,66,45,9,26,39,50,10,16,71,0,19,
# 57,6,23,46,58,24,2,69,30,49,12,20,55,64,17,32,5,48,61,37,18,11,53,
# 38,22,72,67,40,28,47,44,33,63,25,70,36,68,56,42,75,73,65,74,1,29,14,
# 21,52,7,31,62,59,8]

# DATASET_DIRNAME = 'E:/我的雲端硬碟/dataset/mushroom_2020.5.23/NIR/'
DATASET_DIRNAME = 'C:/datasets/'
BIG_IMAGE_FILENAME = 'D:/吳鼎榮/林菇/NIR/採收菇1/MAT/stage1_air_1_NIR_RT.mat'

total_acc = 0
total_loss = 0

K_fold = 11

#
def mat2npy(fullfilename):
    import h5py
    import pathlib
    dirname = os.path.dirname(fullfilename)
    basename = os.path.basename(fullfilename)
    filename, file_extension = os.path.splitext(basename)
    mat = h5py.File(fullfilename)
    mat_t = mat['data']
    # print(mat_t.shape)
    
    if not os.path.exists(dirname + '/' + filename + '/'):
        # os.mkdir(dirname + '/' + filename + '/')
        pathlib.Path(dirname + '/' + filename + '/').mkdir(parents=True, exist_ok=True) 
    np.save(dirname + '/' + filename + '/' + filename + '.npy', mat_t)

#
def npy2pngs(fullfilename):
    from PIL import Image 
    import pathlib
    dirname = os.path.dirname(fullfilename)
    basename = os.path.basename(fullfilename)
    filename, file_extension = os.path.splitext(basename)
    images = np.load(fullfilename)
    band = images.shape[0]
    
    for i in range(band):
        image = np.transpose(images[i])
        rescaled = (255.0 / np.nanmax(image) * (image - np.nanmin(image))).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im = im.convert('RGB') # 從8 bits轉成24 bits
        
        if not os.path.exists(dirname + '/images/'):
            # os.mkdir(dirname + '/images/')
            pathlib.Path(dirname + '/images/').mkdir(parents=True, exist_ok=True) 
        im.save(dirname + '/images/' + filename + '_' + np.str(i) + '.png')

#
def mat2pngs(fullfilename):
    import h5py
    from PIL import Image 
    import pathlib
    dirname = os.path.dirname(fullfilename)
    basename = os.path.basename(fullfilename)
    filename, file_extension = os.path.splitext(basename)
    images = h5py.File(fullfilename)
    images = images['data']
    band = images.shape[0]

    for i in range(band):
        image = np.transpose(images[i])
        rescaled = (255.0 / np.nanmax(image) * (image - np.nanmin(image))).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im = im.convert('RGB') # 從8 bits轉成24 bits
        
        if not os.path.exists(dirname + '/' + filename + '/images/'):
            pathlib.Path(dirname + '/' + filename + '/images/').mkdir(parents=True, exist_ok=True) 
        im.save(dirname + '/' + filename + '/images/' + filename + '_' + np.str(i) + '.png')

# 
def gen_pngs_from_mat(fullfilename): # 雖沒有中間檔，但產生png檔的速度很慢 (一分鐘才產兩個png檔)
    ## Load numpy file and split to image files. This is for checking the images differences among bands in a file
    mat2pngs('D:/吳鼎榮/林菇/VNIR/MAT/stage1_air_2_VNIR_RT.mat')

#
def gen_pngs_from_npy(fullfilename):  # 雖有中間檔，但產生中間檔後再產生png檔的速度就很快(約3分鐘完成616個NIR檔)
    dirname = os.path.dirname(fullfilename)
    basename = os.path.basename(fullfilename)
    filename, _ = os.path.splitext(basename)

    ## Load matlab file and convert to numpy file
    mat2npy(fullfilename) # 中間檔  
    ## Load temporary numpy file and split to image files. This is for checking the images differences among bands.
    npy2pngs(dirname + '/' + filename + '/' + filename + '.npy')
    ## remove temporary file
    os.remove(dirname + '/' + filename + '/' + filename + '.npy')    

# 
def for_check_band_noise():
    big_image_filename = input('Big Image filename [' + BIG_IMAGE_FILENAME + ']: ') or BIG_IMAGE_FILENAME

    print('Starting...')
    gen_pngs_from_npy(big_image_filename)
    # gen_pngs_from_mat(big_image_filename)  # 太慢，不採用

#
def load_matfile(dir, filename):
    import scipy.io as sio
    if filename.endswith('.mat'):
      tmpfile = sio.loadmat(dir + '/' + filename)
      tmpfile = tmpfile['ks']  # ks:對應matlab存檔時用的變數
      return filename, tmpfile

#
def load_npyfile(dir, filename):
    if filename.endswith('.npy'):
      return filename, np.load(os.path.join(dir, filename))

#
def load_pklfile(dir, filename):
    import pickle
    if filename.endswith('.pkl'):
      with open(dir + '/' + filename, 'rb') as fin:
        return pickle.load(fin)

#
def mush_load_model():
    from keras.models import load_model
    global model
    print('my_model.h5 is loading ...')
    model = load_model('my_model.h5')
    print('my_model.h5 was loaded OK.')

#
def mush_save_model():
    print('my_model.h5 is saving ...')
    model.save('my_model.h5') # save model
    print('my_model.h5 was saved OK.')

# 
def mush_load_weights():
    global model
    print('model weights are loading ...')
    model.load_weights('./checkpoints/my_checkpoint')
    print('model weights were loaded OK.')

# 
def mush_save_weights():
    global model
    if 1: # 測試用
        print('Do not save weights.')
    else:
        print('model weights are saving ...')
        model.save_weights('./checkpoints/my_checkpoint')
        print('model weights were saved OK.')

# no use
def adj_fixed2Droi(roi_data):
    band = roi_data.shape[0]  # band, row, col是對應matlab存檔時所用變數的順序
    row = roi_data.shape[1]
    col = roi_data.shape[2]
    roi_data = roi_data.reshape((band, col * row), order='F')
    tmpdata = np.zeros((band, ROI_WIDTH * ROI_HEIGHT))
    tmpdata[:,:col*row] = roi_data    
    # plt.figure()
    # plt.imshow(tmpdata[0])
    # plt.show()
    return tmpdata

#
def adj_fixed3Droi(roi_data):
    band = roi_data.shape[0]  # band, row, col是對應matlab存檔時所用變數的順序
    row = roi_data.shape[1]
    row_half = row // 2        
    col = roi_data.shape[2]
    col_half = col // 2
    WIDTH_HALF = ROI_WIDTH // 2
    HEIGHT_HALF = ROI_HEIGHT // 2
    tmpdata = np.zeros((band, ROI_WIDTH, ROI_HEIGHT)) # np.zeros排法是先col再row
    tmpdata[:, (WIDTH_HALF-row_half):(WIDTH_HALF-row_half)+row, (HEIGHT_HALF-col_half):(HEIGHT_HALF-col_half)+col] = roi_data  # push mushroom in the center
    # plt.figure()
    # plt.imshow(tmpdata[0])
    # plt.show()
    return tmpdata

# no use
def adj_to2d(roi_data):
    band = roi_data.shape[0]  # band, row, col是對應matlab存檔時所用變數的順序
    row = roi_data.shape[1]
    col = roi_data.shape[2]
    return roi_data.reshape((band, col * row), order='F')

#
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(1,5))  # 一列4個
  plt.yticks([])
  thisplot = plt.bar(range(1,5), predictions_array, color="#777777")  # 一列4個
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#
def plot_image(i, predictions_array, true_label, img, basename):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

    '''
   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                 100*np.max(predictions_array),
                                 class_names[true_label]),
                                 color=color)
   '''
  '''
  plt.xlabel("A.{} -> P.{}".format(class_names[true_label],       #改成自己看得懂的方式                          
                                 class_names[predicted_label]),
                                 color=color)
  '''
  plt.xlabel("{}".format(basename), color=color)
  
#
def mush_predict_plot():
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = show_test_images.shape[0]  # test set的數量
    num_cols = 4 # 一列4個菇
    # num_rows = num_images / num_cols # 當test set有35(只顯示32張，8x4)張時會掛掉在這一行
    num_rows = num_images / num_cols + 1
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images): # 一列4個菇和4個直方圖
      plt.subplot(num_rows, 2*num_cols, 2*i+1) # 菇的影像
      plot_image(i, predictions[i], test_labels, show_test_images[i], show_test_basename[i]) # 菇的影像
      plt.subplot(num_rows, 2*num_cols, 2*i+2) # 直方圖
      plot_value_array(i, predictions[i], test_labels) # 直方圖
    plt.tight_layout()
    plt.show()

#
def mush_predict_model():
    global class_names, model, predictions

    ## Make predictions
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(test_images)
        
    # 預測第0個是哪一個stage
    print('predict the index 0 of testing data is \'', class_names[np.argmax(predictions[0])], '\'')
    
    # verify是否為真
    print('verify the index 0 of testing data is \'', class_names[test_labels[0].astype(int)], '\'')
    
    if (np.argmax(predictions[0]) == test_labels[0].astype(int)):
        print('Prediction correct!')
    else:
        print('Prediction inccorect!')

    mush_predict_plot()
    # a = model.predict_classes(test_images) #test

#
def accuracy_chart(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']    
    loss=history.history['loss']
    val_loss=history.history['val_loss']    
    epochs_range = range(EPOCHS)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

#
def extract_features(roi_data, color):
    y=[]
    plot_x = np.arange(852, 1704, 2);  # 參考.hdr檔的波長範圍

    f_mean = np.mean(roi_data, 1);  # (444)
    y.append(f_mean)

    # f_var = np.var(roi_data, 1);    
    # y.append(f_var)

    # f_median = np.median(roi_data, 1);
    # y.append(f_median)
    
    # f_max = np.amax(roi_data, 1);   
    # y.append(f_max)

    # f_min = np.amin(roi_data, 1);   
    # y.append(f_min)

    # plot_y = f_mean
    # plt.plot(plot_x, plot_y, color);    
                                        # median+mean = %
    y = np.array(y)                     # median+var = %
    y = y.reshape((-1))                 # median+mean+var = %
    return y                            # median+mean+var+max = %
                                        # median+mean+var+min = %
                                        # median+mean+max+min = %
                                        # mean+var+min = %
                                        # median+mean+var+max+min = %
                                        
#
def remove_zero_data(roi_data): # 挺費時
    nozero_img = []
    pos = []

    img = roi_data[300];  # (41, 58) 共41*58=2378個座標值
    # band = 444 # 如果mush_read_dataset()同時讀進NIR和VNIR的話，NIR的band是444，VNIR的是449 (可從matlab存檔時做修改)，兩個不一致的話會造成mush_read_dataset()最後跑np.array(train_images)時將二維變成一維array，造成後面的問題
    band = roi_data.shape[0]
    val = filters.threshold_otsu(img)
    mask = img < val
    row = mask.shape[0]
    col = mask.shape[1]
    # plt.imshow(mask.reshape((row,col)), cmap='gray', interpolation='nearest') 
    
    # record the non-zero postion of the mask
    for i in range(row):
        for j in range(col):
            if mask[i][j] != True:  # !=0 的意思
                pos.append((i,j))  # list -> e.g. 1606個non-zero的座標值(x,y)
    
    # copy the value of non-zero postion into new array
    for i in range(band):
        nozero_img.append([],) # 加逗點是增加列，沒有逗點是增加行
        for j in range(len(pos)):  # e.g. 1606個
                row = pos[j][0]
                col = pos[j][1]                
                nozero_img[i].append(roi_data[i][row][col])  # (444, 1606)
    
    nozero_img = np.array(nozero_img)
    return nozero_img

#
def mush_shuffle_dataset():
    global dataset_filestr
    
    stage1_files = [0,'stage1.1.npy'], [0,'stage1.2.npy'], [0,'stage1.3.npy'], [0,'stage1.4.npy'], [0,'stage1.5.npy'], \
                    [0,'stage1.6.npy'], [0,'stage1.7.npy'], [0,'stage1.8.npy'], [0,'stage1.9.npy'], [0,'stage1.10.npy'], \
                    [0,'stage1.11.npy'], [0,'stage1.12.npy'], [0,'stage1.13.npy'], [0,'stage1.14.npy'], [0,'stage1.15.npy'], \
                    [0,'stage1.16.npy'], [0,'stage1.17.npy'], [0,'stage1.18.npy'], [0,'stage1.19.npy'], [0,'stage1.20.npy'], \
                    [0,'stage1.21.npy'], [0,'stage1.22.npy'], [0,'stage1.23.npy'], [0,'stage1.24.npy'], [0,'stage1.25.npy'], \
                    [0,'stage1.26.npy'], [0,'stage1.27.npy'], [0,'stage1.28.npy'], [0,'stage1.29.npy'], [0,'stage1.30.npy'], \
                    [0,'stage1.31.npy'], [0,'stage1.32.npy'], [0,'stage1.33.npy']
    stage2_files = [1,'stage2.1.npy'], [1,'stage2.2.npy'], [1,'stage2.3.npy'], [1,'stage2.4.npy'], [1,'stage2.5.npy'], \
                    [1,'stage2.6.npy'], [1,'stage2.7.npy'], [1,'stage2.8.npy'], [1,'stage2.9.npy'], [1,'stage2.10.npy'], \
                    [1,'stage2.11.npy'], [1,'stage2.12.npy'], [1,'stage2.13.npy'], [1,'stage2.14.npy'], [1,'stage2.15.npy'], \
                    [1,'stage2.16.npy'], [1,'stage2.17.npy'], [1,'stage2.18.npy'], [1,'stage2.19.npy'], [1,'stage2.20.npy'], \
                    [1,'stage2.21.npy'], [1,'stage2.22.npy'], [1,'stage2.23.npy'], [1,'stage2.24.npy'], [1,'stage2.25.npy'], \
                    [1,'stage2.26.npy'], [1,'stage2.27.npy'], [1,'stage2.28.npy'], [1,'stage2.29.npy']
    stage3_files = [2,'stage3.1.npy'], [2,'stage3.2.npy'], [2,'stage3.3.npy'], [2,'stage3.4.npy'], [2,'stage3.5.npy'], \
                    [2,'stage3.6.npy'], [2,'stage3.7.npy'], [2,'stage3.8.npy'], [2,'stage3.9.npy'], [2,'stage3.10.npy'], \
                    [2,'stage3.11.npy'], [2,'stage3.12.npy'], [2,'stage3.13.npy'], [2,'stage3.14.npy'], [2,'stage3.15.npy'], \
                    [2,'stage3.16.npy'], [2,'stage3.17.npy'], [2,'stage3.18.npy'], [2,'stage3.19.npy'], [2,'stage3.20.npy'], \
                    [2,'stage3.21.npy'], [2,'stage3.22.npy'], [2,'stage3.23.npy'], [2,'stage3.24.npy'], [2,'stage3.25.npy'], \
                    [2,'stage3.26.npy'], [2,'stage3.27.npy'], [2,'stage3.28.npy'], [2,'stage3.29.npy']
    stage4_files = [3,'stage4.1.npy'], [3,'stage4.2.npy'], [3,'stage4.3.npy'], [3,'stage4.4.npy'], [3,'stage4.5.npy'], \
                    [3,'stage4.6.npy'], [3,'stage4.7.npy'], [3,'stage4.8.npy'], [3,'stage4.9.npy'], [3,'stage4.10.npy'], \
                    [3,'stage4.11.npy'], [3,'stage4.12.npy'], [3,'stage4.13.npy'], [3,'stage4.14.npy'], [3,'stage4.15.npy'], \
                    [3,'stage4.16.npy'], [3,'stage4.17.npy'], [3,'stage4.18.npy'], [3,'stage4.19.npy'], [3,'stage4.20.npy']
                    
    stage1_files = np.array(stage1_files)
    stage2_files = np.array(stage2_files)
    stage3_files = np.array(stage3_files)
    stage4_files = np.array(stage4_files)
    
    dataset_filestr = np.vstack((stage1_files, stage2_files, stage3_files, stage4_files))

    index = np.array(range(len(dataset_filestr)))
    np.random.shuffle(index)
    dataset_filestr = dataset_filestr[index,:]


#
def mush_read_random_dataset_for_K_foldCV(dirname, Val_foldno):  # used for cross validation
    global train_images, train_labels, test_images, test_labels, show_test_images, show_test_basename
    global dataset_filestr

    color={0:'r', 1:'g', 2:'b', 3:'y'};

    folder_name = [x for x in dirname.split(',')]  # str: 'E:/test' -> ['E:/test/',]
    folder_no = np.shape(folder_name)[0]  # type(np.shape(folder_name)) is tuple, type(np.shape(folder_name)[0]) is int 

    for q in range(folder_no):
        plt.figure        
        dir = folder_name[q]
        for j in range(len(dataset_filestr)):
            
            if j // K_fold != Val_foldno:  # training data
                _, roi_data = load_npyfile(dir, dataset_filestr[j, 1])  # load training data
                print('[{:d}] train filenme='.format(Val_foldno), dataset_filestr[j, 1])
                if roi_data is not None:  # 如果folder中沒有mat檔就會是None
                    roi_data = remove_zero_data(roi_data) # 3維->2維。挺費時
                    feature_data = extract_features(roi_data, color[np.int(dataset_filestr[j,0])])  # (888)
                    train_images.append(feature_data)
                    train_labels.append(dataset_filestr[j,0])
            else:  # testing data
                basename, roi_data = load_npyfile(dir, dataset_filestr[j, 1])  # load testing data
                show_roi_data = roi_data
                print('[{:d}] test filenme='.format(Val_foldno), dataset_filestr[j, 1])
                if roi_data is not None:  # 如果folder中沒有mat檔就會是None
                    roi_data = remove_zero_data(roi_data) # 3維->2維。挺費時
                    feature_data = extract_features(roi_data, color[np.int(dataset_filestr[j,0])])  # (888)
                    test_images.append(feature_data)
                    test_labels.append(dataset_filestr[j,0])
                    # show_test_images is for showing in chart
                    show_roi_data = adj_fixed3Droi(show_roi_data)  # 以最大菇的size來顯示每顆菇
                    show_test_images.append(show_roi_data[300]) 
                    show_test_basename.append(basename)
    
    # convert list to array (each array size in the list must be the same)
    train_images = np.array(train_images)
    train_labels = np.array(np.uint8(train_labels))
    test_images = np.array(test_images)
    test_labels = np.array(test_labels, dtype=np.uint8) # 同train_labels寫法，只是換不同方式寫
    show_test_images = np.array(show_test_images)
    show_test_basename = np.array(show_test_basename)
    plt.show()
    
#
def mush_read_dataset(dirname):
    global train_images, train_labels, test_images, test_labels, show_test_images, show_test_basename

    color={'stage1':'r', 'stage2':'g', 'stage3':'b', 'stage4':'y'};

    folder_name = [x for x in dirname.split(',')]  # str: 'E:/test' -> ['E:/test/',]
    folder_no = np.shape(folder_name)[0]  # type(np.shape(folder_name)) is tuple, type(np.shape(folder_name)[0]) is int 
    
    # load training data
    for q in range(folder_no):
        dirname1 = folder_name[q] + 'train/'
        k = 0
        plt.figure
        for j in 'stage1', 'stage2', 'stage3', 'stage4':
            dir = dirname1 + j
            for i in os.listdir(dir):
                _, roi_data = load_npyfile(dir, i) # Alan
                # _, roi_data = load_matfile(dir, i)
                print('[{:d}] train filenme='.format(q), i)
                if roi_data is not None:  # 如果folder中沒有mat檔就會是None
                    roi_data = remove_zero_data(roi_data) # 3維->2維。挺費時
                    feature_data = extract_features(roi_data, color[j])  # (888)
                    train_images.append(feature_data)
                    train_labels.append(np.str(k))
            k = k + 1
            
    # load testing data
    for q in range(folder_no):
        dirname2 = folder_name[q] + 'test/'
        k = 0
        plt.figure
        for j in 'stage1', 'stage2', 'stage3', 'stage4':
            dir = dirname2 + j
            for i in os.listdir(dir):
                basename, roi_data = load_npyfile(dir, i) # Alan
                # basename, roi_data = load_matfile(dir, i)
                show_roi_data = roi_data
                print('[{:d}] test filenme='.format(q), i)
                if roi_data is not None:
                    roi_data = remove_zero_data(roi_data) # 挺費時
                    feature_data = extract_features(roi_data, color[j])
                    test_images.append(feature_data)
                    test_labels.append(np.str(k))
                    
                    # show_test_images is for showing in chart
                    show_roi_data = adj_fixed3Droi(show_roi_data)  # 以最大菇的size來顯示每顆菇
                    show_test_images.append(show_roi_data[300]) 
                    show_test_basename.append(basename)
            k = k + 1
 
    # convert list to array (each array size in the list must be the same)
    train_images = np.array(train_images)
    train_labels = np.array(np.uint8(train_labels))
    test_images = np.array(test_images)
    test_labels = np.array(test_labels, dtype=np.uint8) # 同train_labels寫法，只是換不同方式寫
    show_test_images = np.array(show_test_images)
    show_test_basename = np.array(show_test_basename)
    plt.show()

#
def mush_train_model(run_no, times):
    global model, train_images, train_labels
    global total_acc, total_loss
    
    # shuffle train_images
    index = range(len(train_images[:,0])) # [76,888] -> range(76)
    index = np.array(index)
    np.random.shuffle(index)
    train_images = train_images[index,:]
    train_labels = train_labels[index]    
    
    ## Train the model
    # Feed the model
    # history = model.fit(train_images, train_labels, batch_size=16, validation_data=(test_images, test_labels), epochs=EPOCHS, verbose=1)
    history = model.fit(train_images, train_labels, batch_size=16, validation_data=(test_images, test_labels), shuffle=True, epochs=EPOCHS, verbose=1)
    # history = model.fit(train_images, train_labels, batch_size=16, validation_split=0.33, epochs=EPOCHS, verbose=1)
    # history = model.fit(train_images, train_labels, batch_size=4, epochs=EPOCHS)
    ## Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)

    total_acc = total_acc + test_acc
    total_loss = total_loss + test_loss
    avg_acc = total_acc / run_no
    avg_loss = total_loss / run_no    
    
    out_str = '\n[{},{}] Test accuracy:{}, loss:{}, avg_acc={}, avg_loss={}\n'.format(times, run_no, test_acc, test_loss, avg_acc, avg_loss)
    print(out_str)
    
    with open('c:/datasets/cross_validation.txt', 'a') as out_file:
        out_file.write(out_str)    
    with open('c:/datasets/cross_validation_detail.txt', 'a') as out_file:
        out_file.write(out_str)    
        out_file.write(np.str(dataset_filestr))

    accuracy_chart(history);

#
def mush_create_model(category):
    global model, w
    input_features = (train_images.shape[1],)  # (74,888) -> (888) # 要加逗點變tuple，否則就是int，會在下面時出錯

    ## Build the model
    # Set up the layers       
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_features),
        keras.layers.Dense(512, activation='relu'),
        # keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(category)
    ])
   
    # Compile the model
    sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])
                  # metrics=['accuracy'])
    
    model.summary()
    w = model.get_weights()

#
def classify_mushroom_stage(train, Val_foldno, times):
    # enable input mutiple dataset folders with comma e.g. e:/dataset1,e:/dataset2
    # dirname = input('Datasets path [' + DATASET_DIRNAME + ']: ') or DATASET_DIRNAME
    dirname = DATASET_DIRNAME

    if train == 'train':
        # mush_read_dataset(dirname)
        mush_read_random_dataset_for_K_foldCV(dirname, Val_foldno)  # K-fold cross validation
        mush_create_model(4) # 分4類
        mush_train_model(Val_foldno + 1, times)  # start with 1
        mush_predict_model()
        # mush_save_weights()
       
    else:  # predict
        # mush_read_dataset(dirname)
        mush_read_random_dataset_for_K_foldCV(dirname, Val_foldno)
        mush_create_model(4) # 分4類
        mush_load_weights()
        mush_predict_model()

#    
if __name__ == '__main__':  
    
    # for_check_band_noise() 
    
    for j in range(13):
        mush_shuffle_dataset()
        for i in range(K_fold):  # run K_fold times
            classify_mushroom_stage('train', i, j)
            # classify_mushroom_stage('predict')

            train_images = []
            train_labels = []
            test_images = []
            test_labels = []
            show_test_images = []
            show_test_basename = []
            
        dataset_filestr = []
    
    print('\nfinished.')



