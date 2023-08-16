import pandas as pd
from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
import glob
import argparse
from tqdm import tqdm
import torch
import sys
import os

sys.path.append('../MarrowScope/HemeFind_scripts/')
from models.models import Myresnext50
from train.train_classification import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import Img_DataLoader
from tqdm import tqdm
### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import albumentations

import numpy as np
parser = argparse.ArgumentParser()


#########################################################

parser.add_argument('--patch_repo_dir', type=str,
                    help='Directory in which the cropped patch are stored')

parser.add_argument('--save_destination', type=str,
                    help='Directory in which you would like to store the computaing results')


parser.add_argument('--save_vis', default=False, type=bool,
                    help='whether also save the visulization')

parser.add_argument('--batch_size', default=32, type=bool,
                    help='whether also save the visulization')

args = parser.parse_args()

cell_types = ['adequate','blood','clot']

cell_types_df = pd.DataFrame(cell_types, columns=['Types'])# converting type of columns to 'category'
cell_types_df['Types'] = cell_types_df['Types'].astype('category')# Assigning numerical values and storing in another column
cell_types_df['Types_Cat'] = cell_types_df['Types'].cat.codes

enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Types_Cat']]).toarray())# merge with main df bridge_df on key values
cell_types_df = cell_types_df.join(enc_df)

image_names  = [x.split('/')[-1] for x in glob.glob(args.patch_repo_dir+'/*')] #
#len(image_names)
x_list = []
y_list = []
for _names in image_names:
    x_list.append(int(int(_names.split('_')[1])/512))
    y_list.append(int(int(_names.split('_')[2].split('.')[0])/512))
    
resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
My_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = 3)

checkpoint_PATH = '../HemeFind/HemeFind_scripts/checkpoints/checkpoint_best_iteration3.ckpt'
checkpoint = torch.load(checkpoint_PATH)

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        
        name = k[7:] # remove `module.`
        
        new_state_dict[name] = v
    
    return new_state_dict

checkpoint  = remove_data_parallel(checkpoint['model_state_dict'])

My_model.load_state_dict(checkpoint, strict=True)

transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(mean=(0.5637, 0.5637, 0.5637), std=(0.2381, 0.2381, 0.2381)),

        ]
    )

from tqdm import tqdm
My_model = My_model.cuda().eval()

Orig_img = Img_DataLoader(img_list= glob.glob(args.patch_repo_dir+'/*'), split='compute',df= cell_types_df,transform = transform_pipeline)
shuffle = False

dataloader = DataLoader(Orig_img, batch_size=args.batch_size, num_workers=2, shuffle=shuffle)


for i, _batch in enumerate(tqdm(dataloader)):
    
    if i == 0:

        images = _batch["image"].cuda()
        #label = _batch["label"]
        x_cor    = [int(int(x.split('/')[-1].split('_')[1])/512) for x in _batch['ID']]
        y_cor    = [int(int(x.split('/')[-1].split('_')[-1].split('.')[0])/512) for x in _batch['ID']]
        pred_prob = My_model(images)

        pred_prob = torch.flatten(pred_prob, start_dim=1).detach().cpu().numpy()
    else:
        images = _batch["image"].cuda()
        #_label = _batch["label"]
        _x_cor    = [int(int(x.split('/')[-1].split('_')[1])/512) for x in _batch['ID']]
        _y_cor    = [int(int(x.split('/')[-1].split('_')[-1].split('.')[0])/512) for x in _batch['ID']]
        
        
        _pred_prob = My_model(images)
        
        _pred_prob = torch.flatten(_pred_prob, start_dim=1).detach().cpu().numpy()
        
        x_cor = x_cor + _x_cor
        y_cor = y_cor + _y_cor
        pred_prob = np.concatenate((pred_prob, _pred_prob))
print('Finished computing the prob logits for the region quality')
        
df_res = pd.DataFrame(data= pred_prob, columns = ['adequate', 'blood', 'clot'])
df_res['x'] = x_cor
df_res['y'] = y_cor
df_res['cord'] = list(zip(x_cor,y_cor))

print('Saving results')
print('Saving the quality score table')
if os.path.exists(f"{args.patch_repo_dir.split('/patches/')[0]}/results/") == False:
    os.mkdir(f"{args.patch_repo_dir.split('/patches/')[0]}/results/")
result_dir = f"{args.patch_repo_dir.split('/patches/')[0]}/results/{args.patch_repo_dir.split('/patches/')[1].split('patches')[0]}slide_res/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

ID = result_dir.split('results')[1].split('/')[-2].split('slide_res')[0]
print(ID)
df_res.to_csv(f"{result_dir}{ID}.csv")
    
    
if args.save_vis:
    print('Saving the visulization as well')
    assert 1==2, "The visulization method is not ready yet！Some errors that I will fix when I have time"
 
    for _category in ['adequate', 'blood', 'clot']:
        #converted_x =
        heatmap = np.zeros((int((max(x_cor)/512))+1,int(max(y_cor)/512)+1))
        for ((x,y),z) in zip(df_res['cord'].tolist(), df_res[_category].tolist()):
            heatmap[x, y] = z
        heatmap = heatmap.T
        #heatmap = 1- heatmap
        #plt.figure(figsize = (max(int(x_cor/512)/100),
        #                         max(int(y_cor/512)/100)))
        plots = sns.heatmap(heatmap, cbar=False, vmax=1, vmin=0, yticklabels=False, xticklabels=False, cmap="Greens")
        plots.savefig(f"{args.patch_repo_dir.split('patches')[0]}slide_res/{ID}_{_category}.png", dpi=400)
    
 
