def multiprocessing():
    '''
    
    '''

'''
The cell classifier model could get the 
'''
import sys
sys.path.append('../HemeYolo/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms

from Classification.Heme_classifer import  DeepHeme
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import os
import glob
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil 
import argparse


import albumentations
from Classification.DataLoader import Img_DataLoader
from Classification.ResNext50 import Myresnext50


def sorting_cells(df_result, img_dir):
    '''
    After the prediction 
    sorting the cells into the 
    celltype folders based on the classes
    '''
    classes = list(set(df_result['celltype']))
    for _class in classes:
        if not os.path.exists(os.path.join(img_dir, _class)):
            os.mkdir(os.path.join(img_dir, _class))
        cell_dir = os.path.join(img_dir, _class)
        df_percell = df_result[df_result['celltype']==_class]
        for _percelldir in df_percell['dir']:
            shutil.move(_percelldir, cell_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--patch_repo_dir', type=str, default= None, 
                    help='Directory in which the cropped patch are stored')

parser.add_argument('--cell_repo_dir', type=str, default= None, 
                    help='Directory in which the cropped cells are stored, in the case when you just want to make the model work directly on the cell crops')

parser.add_argument('--cell_dirs_list', type=str, default= None,)

parser.add_argument('--save_detailed_table_dir', type=str, default= None,)

parser.add_argument('--return_detailed_results', type=str, default= False,)
parser.add_argument('--sort_cells', type=str, default= False,)
args = parser.parse_args()
def model_create(num_classes = 23, path = 'not_existed_path'):

    resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
    My_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = num_classes)

    checkpoint_PATH = path
    checkpoint = torch.load(checkpoint_PATH)

    checkpoint  = remove_data_parallel(checkpoint['model_state_dict'])

    My_model.load_state_dict(checkpoint, strict=True)

    return My_model

from collections import OrderedDict

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`

        new_state_dict[name] = v

    return new_state_dict


### Create the dataframe

def df_create():
    # ------------------------------------------------------------------------------------------------------------
    

    cell_types = ['B1','B2', 'E1', 'E4', 'ER1','ER2','ER3','ER4','ER5','ER6',
                     'L2','L4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                     'MO2','PL2','PL3','U1','U4']
    cell_types.sort()

    cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])# converting type of columns to 'category'
    cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')# Assigning numerical values and storing in another column
    cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes

    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())# merge with main df bridge_df on key values
    cell_types_df = cell_types_df.join(enc_df)
    return cell_types_df, cell_types

def predition(model, test_lists, df, transform, dataloader, return_detailed_results):
    """
    :param model: deep learning model
    :param test_lists: a list of image patches
    :param df: df_celltypes
    :param transform: normalization
    :param dataloader: dataloader
    :return: the lists of labels

    """
    ### make the progress bar with tqdm 

    for i, _batch in enumerate(tqdm(dataloader)):

        if i == 0:

            images = _batch["image"].cuda()
            #print(label)
            #ID    = [x.split('/')[-2]+"_"+x.split('/')[-1] for x in _batch['ID']]
            pred_prob = model(images)
            pred_prob = torch.flatten(pred_prob, start_dim=1).detach().cpu().numpy()
        else:
            images = _batch["image"].cuda()
            _ID    = [x.split('/')[-2]+"_"+x.split('/')[-1] for x in _batch['ID']]
            _pred_prob = model(images)
            _pred_prob = torch.flatten(_pred_prob, start_dim=1).detach().cpu().numpy()
            pred_prob = np.concatenate((pred_prob, _pred_prob))
    label = np.argmax(pred_prob, axis = 1)

    if return_detailed_results:
        df_result = pd.DataFrame()
        df_result['dir'] = test_lists
        df_result['celltype'] = label
        cell_types = ['B1','B2', 'E1', 'E4', 'ER1','ER2','ER3','ER4','ER5','ER6',
                     'L2','L4', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                     'MO2','PL2','PL3','U1','U4']
        
        ### add 23 columns and get the probability for each cell type
        for i in range(23):
            df_result[cell_types[i]] = pred_prob[:,i]
        df_result.to_csv(args.return_detailed_results, sep='\t')
        return list(label), df_result
    else:
        return list(label)

def main(args):
    if args.patch_repo_dir is not None:
        result_dir = f"{args.patch_repo_dir.split('/patches/')[0]}/results/{args.patch_repo_dir.split('/patches/')[1].split('patches')[0]}slide_res/"

        ID = result_dir.split('results')[1].split('/')[-2].split('slide_res')[0]
        
        celldir = f"{result_dir}extracted_cells"
        Image_dirs = glob.glob(celldir+'/*')
    elif args.cell_repo_dir is not None:
        celldir = args.cell_repo_dir
        
        Image_dirs = glob.glob(celldir+'/*')
    ### alternatively, we could use the cell crops list directly
    elif args.cell_dirs_list is not None:
        celldir = args.cell_dirs_list
        try:
            # convert the first column to list
            Image_dirs = pd.read_csv(celldir, sep='\t', header=None)[0].tolist()
        except:
            raise TypeError("The file is not loaded correctly")
        


    else:
        print("Please provide the patch_repo_dir or cell_repo_dir")
        sys.exit()
    
    transform_pipeline = albumentations.Compose(
        [albumentations.Normalize(mean=(0.5594, 0.4984, 0.6937), std=(0.2701, 0.2835, 0.2176)),])
    
    My_model = model_create(num_classes = 23, path = '/home/harry/Documents/HemeYolo/Classification/weights.ckpt')
    cell_types_df, cellnames= df_create()
    My_model = My_model.cuda().eval()
    Orig_img = Img_DataLoader(img_list=Image_dirs, split='viz', df=cell_types_df, transform=transform_pipeline)
    shuffle = False
    dataloader = DataLoader(Orig_img, batch_size=32, num_workers=2, shuffle=shuffle)
    if args.return_detailed_results:
        assert args.save_detailed_table_dir is not None, "Please provide the save_detailed_table_dir"
        labels_numeric, df_result = predition(model=My_model,
                      test_lists=Image_dirs,
                      df=cell_types_df,
                      transform=transform_pipeline,
                      dataloader=dataloader,
                      return_detailed_results = args.return_detailed_results)
    else:
        labels_numeric = predition(model=My_model,
                      test_lists=Image_dirs,
                      df=cell_types_df,
                      transform=transform_pipeline,
                      dataloader=dataloader)
    cellnames = ['B1','B2', 'E1', 'E4', 'ER1','ER2','ER3','ER4','ER5','ER6',
                     'L2','L4', 'M1','M2', 'M3', 'M4', 'M5', 'M6',
                     'MO2','PL2','PL3','U1','U4']
    labels = [cellnames[i] for i in labels_numeric]
    df = pd.DataFrame()
    df['dir'] = Image_dirs
    df['celltype'] = labels
    print('Done with prediction')

    ### save the detailed table
    if args.save_detailed_table_dir is not None:
        df_result.to_csv(args.save_detailed_table_dir, sep='\t')
    else:
        print("Please provide the save_detailed_table_dir")
        sys.exit()
    
    
    if args.sort_cells:
        print("Let's sort them into the different folders")
        sorting_cells(df_result = df, img_dir= celldir)

main(args)

    
    
### save a list as a table 
# df = pd.DataFrame()
# df['dir'] = Image_dirs
### save without the column names
# df.to_csv(f"{result_dir}predictions.tsv", sep='\t', index=False, header=False)

    
    
    
    