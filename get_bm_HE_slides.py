import pandas as pd
import glob 
import numpy as np
import os
import cv2

def get_bm_wsi_list():
    #glob.glob('../temp/touch_prep/bone_marrow/*')
    wsi_list = glob.glob('../temp/touch_prep/bone_marrow/*')
    wsi_list = [wsi.replace('.png', '.ndpi').split('/')[-1] for wsi in wsi_list]
    
    ### convert it to a dataframe
    wsi_df = pd.DataFrame(wsi_list, columns = ['wsi_name'])

    ### save the dataframe
    wsi_df.to_csv('./wsi_list.csv', index = False, sep='\t')

    return wsi_df

### main function ###
if __name__ == '__main__':
    get_bm_wsi_list()



