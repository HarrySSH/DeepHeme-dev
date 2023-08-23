### build a pipeline here to move the images I want to a new folder
### This script will iteratively show image and ask whether the user wants to move the image to a new folder
### The new folder has been provided by the usser with a stinrg
### The user will see a window of the image, and a prompt asking whether to move the image
### The user can press 'y' to move the image, or press 'n' to skip the image
### The user can press 'q' to quit the program
### The user can not press any other key

### import packages
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
import shutil
import argparse

### starting the pipeline
### get the list of images
img_dirs = glob.glob('/Users/ssun2/Documents/DeepHeme/temp/bone_marrow/*.png')

### get the new folder name
folder = '/Users/ssun2/Documents/DeepHeme/temp/touch_prep/bone_marrow'

### iterate through the images
index = 0
while True:
    ### show a progress bar
    print('Progress: {}/{}'.format(index, len(img_dirs)))
    ### check if the index is out of range
    if index >= len(img_dirs):
        print('You have reached the end of the image list')
        break
    ### read the image
    img = cv2.imread(img_dirs[index])
    ### show the image
    cv2.imshow('image', img)
    ### ask the user whether to move the image
    k = cv2.waitKey(0)
    ### if the user press 'y', move the image to the new folder
    
    if k == ord('y'):
        cv2.destroyAllWindows()
        shutil.move(img_dirs[index], folder)
        index+=1
        # close the window
        
    ### if the user press 'n', skip the image
    elif k == ord('n'):
        cv2.destroyAllWindows()
        index+=1
        
    ### if the user press 'q', quit the program
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
    ### if the user press any other key, tell the user that the key is not valid
    else:
        print('The key you pressed is not valid, please press y, n, or q')
        ### return to the previous image
        
