### load basic packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


###load self written functions
from quality_check import get_top_view_preselection_mask, is_touch_prep

### access the directory of the slides
slide_dirs = glob.glob('/Users/ssun2/Documents/DeepHeme/topview/*')

### iterate through the slides, and get the toouch prep slides
touch_prep_slides = []
for slide_dir in tqdm(slide_dirs):
    ### read the slide_dir into a BGR image
    slide = cv2.imread(slide_dir)
    ### check if the slide is a touch prep slide
    if is_touch_prep(slide):
        touch_prep_slides.append(slide_dir)

### create a temp folder and store the copy of touch prep slides
temp_dir = '/Users/ssun2/Documents/DeepHeme/temp'
### creat the touch prep folder if it does not exist
touch_prep_dir = '/Users/ssun2/Documents/DeepHeme/temp/touch_prep'
bm_dir = '/Users/ssun2/Documents/DeepHeme/temp/bone_marrow'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
if not os.path.exists(touch_prep_dir):
    os.makedirs(touch_prep_dir)
if not os.path.exists(bm_dir):
    os.makedirs(bm_dir)
### copy the touch prep slides to the temp touch prep folder
for slide_dir in tqdm(touch_prep_slides):
    slide_name = slide_dir.split('/')[-1]
    slide_name = slide_name.split('.')[0]
    slide_name = slide_name + '.png'
    slide_path = os.path.join(touch_prep_dir, slide_name)
    slide = cv2.imread(slide_dir)
    cv2.imwrite(slide_path, slide)

### copy the non touch prep slides to the BM folder
for slide_dir in tqdm(slide_dirs):
    if slide_dir not in touch_prep_slides:
        slide_name = slide_dir.split('/')[-1]
        slide_name = slide_name.split('.png')[0]
        slide_name = slide_name + '.png'
        slide_path = os.path.join(bm_dir, slide_name)
        slide = cv2.imread(slide_dir)
        cv2.imwrite(slide_path, slide)




