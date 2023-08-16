import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import os
import openslide
import xml.etree.cElementTree as ET
import random
import os 
import glob
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()


#########################################################

parser.add_argument('--wsi_dir', type=str,
                    help='Directory in which the WSIs that you want to patch are stored')

parser.add_argument('--wsi_name', type=str,
                    help='Name of the wsi slide')

parser.add_argument('--save_destination', type=str,
                    help='Directory in which you would like to store the resulting patches')


parser.add_argument('--patch_size', default=512, type=int,
                    help='The size of the patches')



args = parser.parse_args()

def generate_patches(slidepath,slidename,  patch_size,
                     savepath):
  
    slide_path = os.path.join(slidepath, slidename)
    
    Slide = openslide.OpenSlide(slide_path)
    
    length, width = Slide.level_dimensions[0]

    for _x in tqdm(range(0,length, patch_size)):
        for _y in range(0, width, patch_size):
            patching = Slide.read_region((_x, _y), 0, (patch_size, patch_size))   
            if not os.path.exists(f"{savepath}/{slidename.split('.ndpi')[0]}patches"):
                #print(f"{savepath}/{slidename.split('.ndpi')[0]}patches")
                #assert 1 ==2, 'stop'
                os.mkdir(f"{savepath}/{slidename.split('.ndpi')[0]}patches")
            patching = cv2.cvtColor(np.array(patching), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savepath}/{slidename.split('.ndpi')[0]}patches/patch_{str(_x)}_{str(_y)}.png", patching)

if __name__ == '__main__':

    generate_patches(slidepath=args.wsi_dir,
                     slidename=args.wsi_name,
                     patch_size=args.patch_size,
        
                     savepath = args.save_destination,
                     )


