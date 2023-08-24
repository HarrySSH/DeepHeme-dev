import pandas as pd
from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
import glob
import argparse
from tqdm import tqdm
import torch
import sys
import os
from typing import Dict

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


### We want to use Ray library for multi-processing and distributed computing
import ray
from ray.train.torch import TorchPredictor

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



from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        
        name = k[7:] # remove `module.`
        
        new_state_dict[name] = v
    
    return new_state_dict

checkpoint_PATH = '../HemeFind/HemeFind_scripts/checkpoints/checkpoint_best_iteration3.ckpt'
checkpoint = torch.load(checkpoint_PATH)
checkpoint  = remove_data_parallel(checkpoint['model_state_dict'])

### load checkpoints
My_model.load_state_dict(checkpoint)

ds = ray.data.read_images(args.patch_repo_dir,
                          mode="RGB")

class ResnextModel:
    def __init__(self):
        self.labels = ["adequate", "blood", "clot"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = My_model
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [
                self.labels[i] for i in predicted_classes
            ]
            return {
                "predicted_label": predicted_labels,
                "original_image": batch["original_image"],
                "predicted_prob": prediction.detach().cpu().numpy()
            }
        
 
from tqdm import tqdm

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5637, 0.5637, 0.5637), std=(0.2381, 0.2381, 0.2381)),

        ]
    )

def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "original_image": row["image"],
        "path": row["path"],
        "transformed_image": transform(row["image"]),
    }

transformed_ds = ds.map(preprocess_image)
deepheme_base = ResnextModel()
predictions = transformed_ds.map_batches(
    deepheme_base,
    compute=ray.data.ActorPoolStrategy(
        size=3
    ),  # Use 4 GPUs. Change this number based on the number of GPUs in your cluster.
    num_gpus=1,  # Specify 1 GPU per model replica.
    batch_size=256,  # Use the largest batch size that can fit on our GPUs
)

predictions.drop_columns(["original_image"]).write_parquet(f"{args.save_destination}/predictions.parquet")
print(f"Predictions saved to {args.save_destination}/predictions.parquet")

print('Finished computing the prob logits for the region quality')
 