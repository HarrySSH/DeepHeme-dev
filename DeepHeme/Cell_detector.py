import sys
sys.path.append('../HemeYolo/')
from pytorchyolo import detect, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
# Load the YOLO model
model = models.load_model(
  "../HemeYolo/yolov3-custom.cfg", 
  "../HemeYolo/ckpt/checkpoints512.pth")
import glob
from Classification.Heme_classifer import DeepHeme_v2, DeepHeme
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
from Visualization.Draw_box_dots import drawer
import seaborn as sns
import random

import numpy as np
import os


def extract_cell_patches(boxes, save_dir,patchID, img):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    coordinates_list = []
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    for x1,y1,x2,y2 in boxes[:,:4]:
        x_center = int(np.round((x1+x2)/2))
        y_center = int(np.round((y1 + y2) / 2))
        
        if x_center-48<0 or y_center-48<0 or x_center+48>=img.shape[1] or y_center+48>=img.shape[0]:
            #canvas = np.zeros((255,255,3))
            pass
            
        else:
            patch = img[y_center-48:y_center+48,
                    x_center-48:x_center+48]
            #patch = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            patch = Image.fromarray(patch)
            patch_name = patchID+'_x'+str(x_center)+'_y'+str(y_center)+'.png'

            patch.save(os.path.join(save_dir, patch_name))

            #name_dic[patch_name] = os.path.join(save_dir, patch_name)
            coordinates_list.append((x_center,y_center))
            x1_list.append(int(x1))
            x2_list.append(int(x2))
            y1_list.append(int(y1))
            y2_list.append(int(y2))

parser = argparse.ArgumentParser()
parser.add_argument('--patch_repo_dir', type=str,
                    help='Directory in which the cropped patch are stored')
parser.add_argument('--max_patch_number', type=int,
                    default = 300,
                    help='How many patches you want to run yolo on')
parser.add_argument('--image_quality_score', type=int,
                    default = 0.8,
                    help='How many patches you want to run yolo on')

args = parser.parse_args()

def main(args):
    result_dir = f"{args.patch_repo_dir.split('/patches/')[0]}/results/{args.patch_repo_dir.split('/patches/')[1].split('patches')[0]}slide_res/"

    ID = result_dir.split('results')[1].split('/')[-2].split('slide_res')[0]
    print('Loading the quality score from previous results')
    try:
        df_res = pd.read_csv(f"{result_dir}predictions.tsv", sep='\t', index_col = 0)
    except:
        raise TypeError("The file is not loaded correctly")
    df_res = df_res.sort_values(['adequate_prob'], ascending=False)
    df_res.index = list(range(df_res.shape[0]))
    
    good_regions=df_res[df_res['adequate_prob']>args.image_quality_score]
    if good_regions.shape[0]>args.max_patch_number:
        print(f'There are many good regions, we subset {args.max_patch_number} regions with the best confidence. For quality and speed')
        good_regions = good_regions.head(args.max_patch_number)
    
    
    p = 0
    if not os.path.exists(result_dir):
        raise TypeError("The fold that contain results does not exist")
    else:
        rootdir = result_dir #+ 'slide_res'
        savedir = os.path.join(rootdir, 'extracted_cells')
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    
    for  path in tqdm(good_regions['path'].tolist()):

        
        image = cv2.imread(path)
        #image = cv2.imread(_dir)
        # Convert OpenCV bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        boxes = detect.detect_image(model, image, conf_thres=0.1, nms_thres=0.1)
        
        patchID = path.split('/')[-1].split('.png')[0]
        
        extract_cell_patches(boxes, savedir,patchID, image)
        
        
        
if __name__ == "__main__":
    main(args)
    print('Done')
    