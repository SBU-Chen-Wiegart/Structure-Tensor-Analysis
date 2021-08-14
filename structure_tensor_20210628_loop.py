# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:32:55 2021

@author: Chen-Wiegart
"""

import matplotlib.pyplot as plt
# import tomopy
import numpy as np
#from PIL import Image 
from skimage import io
from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
import os
import time
from structure_tensor import eig_special_2d, structure_tensor_2d
from skimage import feature
from scipy.ndimage import uniform_filter, generic_filter
# from cupyx.scipy.ndimage import uniform_filter, generic_filter
# import cupy as cp


# %matplotlib qt
# %matplotlib notebook
# %matplotlib inline

'''  Img Shape of each data set
SP pristine: (1124, 395, 633)
CNT pristine: (1081, 293, 774)
SP 200 cycles: (562, 428, 1161)
CNT 200 cycles: (744, 346, 1338)
'''

plt.close('all')
sample = ['S0', 'C0', 'S200', 'C200']
# spl_idx = 2
# volume_size = 160000
volume_size = [3, 80000, 120000, 160000]
filter_size = 50  ## in pixel
# watershed_dir = '/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/watershed_20210615/' + sample[spl_idx] + '/'
# in_dir = '/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/watershed_20210615/' + sample[spl_idx] + '/structure_tensor_20210624/'
# out_dir = watershed_dir + 'structure_tensor_20210628/'
# fn = sample[spl_idx] + f'_watershed_threshold_{volume_size}_edge_small.tiff'
# fn_watershed = sample[spl_idx] + '_watershed_wcracks_32bit.tiff'

t0 = time.time()
for k in sample:
    for l in volume_size:
        plt.close('all')
        watershed_dir = '/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/watershed_20210615/' + k + '/'
        out_dir = watershed_dir + 'structure_tensor_20210628/'
        fn_watershed = f'{k}_watershed_threshold_{l}.tiff'
        # small_eigen = np.float32(io.imread(in_dir+fn))
        # plt.figure()
        # plt.imshow(small_eigen[270])

        '''
        Turn watershed into binary image
        '''
        img = np.float32(io.imread(watershed_dir+fn_watershed))
        img[img != 0] = 255
        # img[img > 1] = 255
        plt.figure()
        plt.imshow(img[0])
        # # fn_out = fn[:-5] + '_binary.tiff'
        # # io.imsave(out_dir+fn_out, np.float32(img))
        
        # '''
        # Plot histogram and Segment data
        # '''
        # # flat = img.flatten()
        # # hist, bins = np.histogram(flat, bins = 256)
        # # plt.figure()
        # # plt.hist(flat, bins=256)
        # # plt.figure()
        # # plt.plot(bins[:-1], hist)
        # # peaks, pi = find_peaks(hist,height=100000)
        # # counts = pi['peak_heights']
        # # plt.plot(bins[peaks], counts, 'rx')
        # # his_btw2peaks = hist[peaks[-2]:peaks[-1]]    #y: counts
        # # bins_btw2peaks = bins[peaks[-2]:peaks[-1]]   #x: gary scale
        # # his_btw2peaks_min = np.amin(his_btw2peaks)
        # # bins_btw2peaks_index = np.where(his_btw2peaks == his_btw2peaks_min)
        # # threshold = bins_btw2peaks[bins_btw2peaks_index]
        # # plt.plot(bins_btw2peaks[bins_btw2peaks_index], his_btw2peaks[bins_btw2peaks_index], 'gp')
        # # # img[img<float(threshold)] = 0
        # # # img_seg = np.where(img < float(threshold), 0, 255)
        # # img = np.where(img < float(threshold), 0, 255)
        # # plt.figure()
        # # plt.imshow(img[270])
        # # fn_out = fn[:-4] + '_seg.tif'
        # # out_dir = in_dir
        # # io.imsave(out_dir+fn_out, np.float32(img))
        
        
        
        '''
        Canny edge detection
        '''
        # a = img[270]
        sigma = 3
        # edge = feature.canny(a, sigma=sigma)
        # plt.figure()
        # plt.imshow(edge)
        
        edge_img = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        
        for i in range(img.shape[0]):
            edge = feature.canny(img[i], sigma=sigma)
            edge_img[i] = edge
            print(f'Canny edge detection of img[{i}/{img.shape[0]}] is done.')
        
        plt.figure()
        plt.imshow(edge_img[270])
        fn_out = fn_watershed[:-5] + '_edge.tiff'
        io.imsave(out_dir+fn_out, np.float32(edge_img))
        
        
        '''
        Structure Tensor
        '''
        sigma = 3
        rho = 5
        
        small_eigen = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        large_eigen = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        
        for j in range(img.shape[0]):
            s = structure_tensor_2d(edge_img[j], sigma, rho)
            val, _ = eig_special_2d(s)
            small_eigen[j] = val[0]
            large_eigen[j] = val[1]
            print(f'Structure tensor of img[{j}/{img.shape[0]}] is done.')
        
        plt.figure()
        plt.imshow(small_eigen[270])
        plt.figure()
        plt.imshow(large_eigen[270])
        fn_small = fn_watershed[:-5] + '_edge_small.tiff'
        io.imsave(out_dir+fn_small, np.float32(small_eigen))
        # fn_large = sample[spl_idx] + '_' + fn[:-5] + '_edge_large.tiff'
        # io.imsave(out_dir+fn_large, np.float32(large_eigen))
        
        
        
        '''
        Apply 3D Mean Filter
        '''
        size = (filter_size, filter_size, filter_size)   # X by Y by Z pixels
        # small_eigen[small_eigen == 0] = np.nan
        # with cp.cuda.Device(0):
        #     small_eigen = cp.asarray(small_eigen)
        t1 = time.time()
        print(f'Applying 3D mean filter to {fn_small}...')
        filter_img = uniform_filter(small_eigen, size=size, mode='constant', cval=0)
        # filter_img = generic_filter(small_eigen, np.nanmean, size=size, mode='constant', cval=np.nan)
        print('3D mean filter to small eigenvalues is done!')
        t2 = time.time()
        mean_filter_time = (t2-t1)
        print(f'Time for 3D mean filter: {mean_filter_time} seconds.')
        plt.figure()
        plt.imshow(filter_img[270])
        fn_filter = fn_watershed[:-5] + f'_edge_small_{filter_size}pix.tiff'
        # fn_filter = fn[:-5] + f'_{filter_size}pix.tiff'
        io.imsave(out_dir+fn_filter, np.float32(filter_img))
        
        
        
        '''
        Edge Gradient
        '''
        # img[img==0] = np.nan
        gradient_img = filter_img * img
        plt.figure()
        plt.imshow(gradient_img[270])
        fn_filter = fn_watershed[:-5] + f'_edge_small_{filter_size}pix_grad.tiff'
        # fn_filter = fn[:-5] + f'_{filter_size}pix_grad.tiff'
        io.imsave(out_dir+fn_filter, np.float32(gradient_img))
        
        
        plt.show()

t3 = time.time()
total_time = (t3-t0)/60
print(f'Total Time for calculating 16 strucutre tensors is {total_time} minutes.')
