import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import math
import time
import pandas as pd
import skimage
import skimage.io
import skimage.transform
import skimage.color
from tqdm import tqdm
from multiprocessing import Pool  
from functools import partial 

### ray is a library for parallel computing
### Neo use this!


### this is a function that leverage the low resolution mask to guide patching the high resolution image


class patching_based_on_mask():
    def __init__(self, low_res_mask, high_res_image_ndpi, patch_size, overlap_threshold, savepath, slidename):
        '''
        low_res_mask: the low resolution mask
        high_res_image_ndpi: the high resolution image
        patch_size: the patch size
        overlap_threshold: the threshold of the overlap between the patch and the mask
        '''

        self.low_res_mask = low_res_mask
        self.high_res_image_ndpi = high_res_image_ndpi
        self.patch_size = patch_size
        self.overlap_threshold = overlap_threshold
        self.savepath = savepath
        self.slidename = slidename

    def save_patch(self, args):
        _x, _y = args
        patching = self.high_res_image_ndpi.read_region((int(_x/self.patch_size), int(_y/self.patch_size)), 0, (self.patch_size, self.patch_size))
        patching = cv2.cvtColor(np.array(patching), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{self.savepath}/{self.slidename.split('.ndpi')[0]}patches/patch_{str(_x)}_{str(_y)}.png", patching)



    #### this is a function that patch the high resolution image based on the low resolution mask
    def patch_high_res_image(self, save = False, number_of_cpu_cores = 1):
        ### get the shape of the high resolution image
        ### the image is readed by openslide,
        ### so the shape is (height, width, channel)
        high_res_image_shape  = self.high_res_image_ndpi.level_dimensions[0]
        ### reshape the low resolution mask to the same shape as the high resolution image
        low_res_mask = cv2.resize(self.low_res_mask, (high_res_image_shape[1], high_res_image_shape[0]))
        
        # change the mask from (255,255,255) to (1)
        low_res_mask = low_res_mask[:,:,0]
        low_res_mask[low_res_mask>0] = 1

        ### checking the overlap between the patch and the mask
        ### if the overlap is larger than the threshold, then the patch is valid
        ### otherwise, the patch is invalid

        ### get the number of patches in the height direction
        num_patch_height = int(math.ceil(high_res_image_shape[0]/self.patch_size))
        ### get the number of patches in the width direction
        num_patch_width = int(math.ceil(high_res_image_shape[1]/self.patch_size))

        ### initialize the patch mask
        patch_mask = np.zeros((num_patch_height, num_patch_width))

        ### compute the overlap between the patch and the mask
  
        # initialize the patch mask  
        patch_mask = np.zeros((num_patch_height, num_patch_width))  
        
        # compute the overlap between the patch and the mask using a sliding window approach  
        stride = self.patch_size  
        window_shape = (self.patch_size, self.patch_size)  
        num_slices = tuple((sz - ws + 1) // stride for sz, ws in zip(low_res_mask.shape, window_shape))  
        strided_low_res_mask = np.lib.stride_tricks.sliding_window_view(low_res_mask, window_shape)  
        
        # calculate the sum over each patch  
        patch_sums = np.sum(strided_low_res_mask, axis=(-1, -2))  
        
        # create a mask where the overlap is larger than the threshold  
        patch_mask = (patch_sums / (self.patch_size * self.patch_size)) > self.overlap_threshold

        ### get the valid patch index
        # get the valid patch index in the height direction
        # get the valid patch index in the width direction
        valid_patch_index_height, valid_patch_index_width = np.where(patch_mask==True) 

        if not os.path.exists(f"{self.savepath}/{self.slidename.split('.ndpi')[0]}patches"):
            #print(f"{savepath}/{slidename.split('.ndpi')[0]}patches")
            #assert 1 ==2, 'stop'
            os.mkdir(f"{self.savepath}/{self.slidename.split('.ndpi')[0]}patches")  

        ### get the valid patch
        # initialize the valid patch
        # Create a list of arguments (_x, _y) for valid patches  
        valid_patches = list(zip(valid_patch_index_height, valid_patch_index_width))  
        if save:  
            # Set the number of processes (use the number of CPU cores for optimal performance)  
            num_processes = os.cpu_count()  
            # warning if the number of processes is larger than the  maximum number of cpu cores
            if num_processes > number_of_cpu_cores:
                print(f"Warning: the number of processes {num_processes} is larger than the maximum number of cpu cores {number_of_cpu_cores}")
                ### set the number of processes to the maximum number of cpu cores
                num_processes = number_of_cpu_cores
                print(f"Set the number of processes to {num_processes}")

                
            # Create a Pool and execute the save_patch function in parallel  
            with Pool(processes=num_processes) as pool:  
                func = partial()
                pool.map(func, valid_patches)  



