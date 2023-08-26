import pandas as pd
from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
import glob
import argparse
from tqdm import tqdm
import torch
import sys
import os
from typing import Dict
import torch
import torch.nn as nn
import time
import re
sys.path.append('../MarrowScope/HemeFind_scripts/')

class Myresnext50(nn.Module):
    def __init__(self, my_pretrained_model, num_classes = 3):
        super(Myresnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, num_classes))
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        
        pred = torch.sigmoid(x.reshape(x.shape[0], 1,self.num_classes))
        return pred
    
#from  import Myresnext50
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

import pyarrow.parquet as pq 


### We want to use Ray library for multi-processing and distributed computing
import ray

ray.init(  
    runtime_env={  
        "working_dir": "..",  # This should be the root directory of your project  
        "pip": ["/home/harry/Documents/DeepHeme-dev/packages/my_project-0.1.0-py3-none-any.whl"],  # Additional packages to install
    }  
) 

print('Why is this not working?')

from ray.train.torch import TorchPredictor

import numpy as np
parser = argparse.ArgumentParser()

### calculate the duration of the whole process
start_time = time.time()
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


    
resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
My_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = 3)



from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        
        name = k[7:] # remove `module.`
        
        new_state_dict[name] = v
    
    return new_state_dict

checkpoint_PATH = '../MarrowScope/HemeFind_scripts/checkpoints/checkpoint_best_iteration3.ckpt'
checkpoint = torch.load(checkpoint_PATH)
checkpoint  = remove_data_parallel(checkpoint['model_state_dict'])

### load checkpoints
My_model.load_state_dict(checkpoint)

ds = ray.data.read_images(args.patch_repo_dir,
                          mode="RGB", include_paths=True)

class ResnextModel:
    def __init__(self):
        self.labels = ["adequate", "blood", "clot"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = My_model.to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():

            prediction = self.model(torch_batch)

            predicted_classes = prediction.argmax(dim=2).detach().cpu()
            
            return {
                "predicted_label": predicted_classes,
                "original_image": batch["original_image"],
                "predicted_prob": prediction.detach().cpu().numpy(),
                'path': batch['path'],
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
    ),  # Use 3 GPUs. Change this number based on the number of GPUs in your cluster.
    num_gpus=1,  # Specify 1 GPU per model replica.
    batch_size=256,  # Use the largest batch size that can fit on our GPUs
)
print('Finished computing the prob logits for the region quality')
if os.path.exists(f"{args.patch_repo_dir.split('/patches/')[0]}/results/") == False:
    os.mkdir(f"{args.patch_repo_dir.split('/patches/')[0]}/results/")

result_dir = f"{args.patch_repo_dir.split('/patches/')[0]}/results/{args.patch_repo_dir.split('/patches/')[1].split('patches')[0]}slide_res/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

predictions.drop_columns(["original_image"])
print('Saving results')
#pq.write_table(predictions, f"{result_dir}predictions.parquet") wrong codes

dfs = []  
for batch in predictions.iter_batches(batch_format="pandas"):  
    dfs.append(batch)  
  
result_df = pd.concat(dfs, axis=0, ignore_index=True)  
del result_df["original_image"]

result_df['predicted_label'] = result_df['predicted_label'].str[1]

### convert the prob logits to prob for each class format as string "[[0.5880363  0.39977467 0.02685326]]"
### first remove the quotes ""
import re  
  
def string_to_list(string):  
    # Remove outer double quotes if present  
    string = string.strip('"')  
  
    # Remove the outer square brackets and split the string by spaces  
    string = string[1:-1].split() [:3]

  
    # Extract the numbers from the string using regular expressions  
    numbers = [float(re.findall(r'\d+\.\d+', num)[0]) for num in string]  
    
  
    return [numbers]

prob_list = []
for x in result_df['predicted_prob'].tolist():
    prob_list.append(string_to_list(x)[0])


result_df['predicted_prob'] = prob_list

result_df['adequate_prob'] = [x[0] for x in result_df['predicted_prob']]
result_df['blood_prob'] = [x[1] for x in result_df['predicted_prob']]
result_df['clot_prob'] = [x[2] for x in result_df['predicted_prob']]
### save the results
result_df.to_csv(f"{result_dir}predictions.tsv", sep='\t', index=False)


#predictions.drop_columns(["original_image"]).write_parquet(f"{result_dir}predictions.parquet")
### also drop the column when the images is not predicted as adequate
#predictions[predictions["predicted_label"] != "adequate"].drop_columns(["original_image"]).write_parquet(f"{result_dir}predictions_no_adequate.parquet")


print(f"Predictions saved to {result_dir}predictions.parquet")

print('Finished computing the prob logits for the region quality')
print('SHUTTING DOWN RAY')

ray.shutdown()

### calculate the duration of the whole process
end_time = time.time()
print('The total time for computing the region quality is {} seconds'.format(end_time - start_time))
 
