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

from concurrent.futures import ThreadPoolExecutor 
from concurrent.futures import ThreadPoolExecutor, as_completed  

### import self built packages
from quality_check import get_top_view_preselection_mask
from utils.image_basics import convert_4_to_3

### ray is a library for parallel computing
### Neo use this!

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

    #### this is a function that patch the high resolution image based on the low resolution mask
    def patch_high_res_image(self, save = False, num_processes = 1):
        
        ### get the shape of the high resolution image
        ### the image is readed by openslide,
        ### so the shape is (height, width, channel)
        high_res_image_shape  = self.high_res_image_ndpi.level_dimensions[0]
        
        ### reshape the low resolution mask to the same shape as the high resolution image
        #low_res_mask = cv2.resize(self.low_res_mask, (high_res_image_shape[1], high_res_image_shape[0]))
        
        
        ### get the largest dimension of the openslide object
        WSI_shape = self.high_res_image_ndpi.level_dimensions[0]
        

        # change the mask from (255,255,255) to (1)
        #assert 1==2, 'stop here for now'
        low_res_mask = self.low_res_mask

        if len(low_res_mask) == 3:
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
        ratio = int(WSI_shape[0]/low_res_mask.shape[1])
        

   
        # compute the overlap between the patch and the mask using a sliding window approach  
        stride = self.patch_size  
        window_shape = (int(self.patch_size/ratio), int(self.patch_size/ratio))

        ### iteration through the low resolution mask
        ### if the non-zero value is more than the thresdhold, then the patch is valid
        ### otherwise, the patch is invalid, that woule be represented in the patch_mask by zero and one
        ### zero means invalid, one means valid
        for i in tqdm(range(0, low_res_mask.shape[0] - window_shape[0] + 1, window_shape[0])):
            for j in range(0, low_res_mask.shape[1] - window_shape[1] + 1, window_shape[1]):
                ### get the patch
                
                patch = low_res_mask[i:i+window_shape[0], j:j+window_shape[1]]
                
                ### check the dimension of the patch
                ### get the number of non-zero elements
                num_non_zero = np.count_nonzero(patch)
                
                ### if the number of non-zero elements is more than the threshold, then the patch is valid
                if num_non_zero > window_shape[0]*window_shape[1]*self.overlap_threshold:
                    ### set the value of the patch mask to be 1
                    patch_mask[int(j/window_shape[0]), int(i/window_shape[1])] = 1
        ### 
        

         
    
        ### show the patch mask
        ### convert the patch_mask to uint8
        #patch_mask_viz = (patch_mask*255).astype(np.uint8)

        #plt.imshow(patch_mask_viz)
        #plt.show()

        #assert 1==2, 'stop here for now'

        # the x and y coordinates of the patch are flipped, so I transpose the patch_mask
        patch_mask = patch_mask.T
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
        valid_patches = list(zip(valid_patch_index_height*512, valid_patch_index_width*512)) 

        return valid_patches 
# main function
if __name__ == '__main__':
    ### use openslide to read the high resolution image
    high_res_image_ndpi = openslide.OpenSlide('/Users/ssun2/Documents/DeepHeme/H17-6116_S10_MSK6_2023-04-27_18.30.44.ndpi')
    ### get the levels of openslide object
    _level = high_res_image_ndpi.level_count - 1
    ### get the lowest resolution of the high resolution image
    low_res_image = high_res_image_ndpi.read_region((0,0), _level, high_res_image_ndpi.level_dimensions[-1])

    ### show the low resolution image
    show_flag = False
    if show_flag:
        plt.imshow(low_res_image)
        plt.show()
    low_res_image = convert_4_to_3(np.array(low_res_image))

    #plt.imshow(low_res_image)
        
    print('The shape of the low resolution image is {}'.format(low_res_image.shape))

    mask = get_top_view_preselection_mask(low_res_image, RGB=False, verbose=False)

    # display the mask
    print('The shape of the mask is {}'.format(mask.shape))

    patching_machine = patching_based_on_mask(low_res_mask=mask, 
                                              high_res_image_ndpi=high_res_image_ndpi, 
                                              patch_size=512, 
                                              overlap_threshold=0.5, 
                                              savepath='/Users/ssun2/Documents/DeepHeme/patches', 
                                              slidename='H17-6116_S10_MSK6_2023-04-27_18.30.44.ndpi')
    
    #patching_machine.patch_high_res_image(save=True,num_processes=1)



    valid_patches = patching_machine.patch_high_res_image(save=True,num_processes=1)

    def save_patch( _x, _y):
        patch_size = patching_machine.patch_size
                

        ### get the patch
        patching = patching_machine.high_res_image_ndpi.read_region((_x, _y), 0, (patch_size, patch_size))
        patching = cv2.cvtColor(np.array(patching), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{patching_machine.savepath}/{patching_machine.slidename.split('.ndpi')[0]}patches/patch_{str(_x)}_{str(_y)}.png", patching)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_patch, x, y) for x, y in valid_patches] 
        for future in tqdm(as_completed(futures), total=len(valid_patches), desc="Processing patches"):  
                pass  # You can retrieve the result here if needed using future.result()  
    


    



