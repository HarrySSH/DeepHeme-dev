import sys
import os
import pandas as pd
import numpy as np
import tqdm
import shutil
import glob
import argparse


tsv = pd.read_excel(open('/media/hdd4/harry/metadata/Slide_Scanning_Tracker.xlsx', 'rb'),)

def renamefunction(_slide):
    if len(_slide.split(' - '))==2:
        
        string = _slide.split(' - ')[0].replace(';','_')
        date = _slide.split(' - ')[1].split(' ')[0]
        time = _slide.split(' - ')[1].split(' ')[1].split('.ndpi')[0]
        convert = string+'_'+date+'_'+time+'.ndpi'
        return convert
    else:
        assert 1==2, 'the name is not what we wanted' 

def get_image_files(latest = False):
    assert 1==2, 'This function is not ready to use yet'
    
    # Replace 'your_folder_path' with the actual path to the folder containing the images
    folder_path = '/media/hdd4/harry/Slides_repo/Plasma_cel_myeloma/slides'
    
    
    if latest:
        image_files = []
        table = pd.read_csv('/media/hdd4/harry/Slides_repo/Plasma_cel_myeloma/table.csv')
        table['caseID'] = table['new_name'].apply(lambda x: x.split('_')[0]+'_'+x.split('_')[1]+'_'+x.split('_')[2])
        lists = []
        for _group in table.groupby('caseID'):
            image_files.append(_group[1].sort_values(['date', 'time_of_scan'], ascending=False).iloc[0]['new_name'])
    else:
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.ndpi') ]
    patch_path = folder_path.replace('slides','patches')
    return [(patch_path, folder_path, image_file) for image_file in image_files]


def read_simple_file(file_dir, columns = ['specnum_formatted','accession_date','part_description']):
    csv = pd.read_csv(file_dir, index_col= 0)
    csv = csv[columns]
    return csv


parser = argparse.ArgumentParser()

parser.add_argument('--General_Dx', type=str,
                    help='Directory in which the cropped patch are stored')

parser.add_argument('--move', type=bool, default=False,
                    help='If True, the patches will be moved to the new directory')

args = parser.parse_args()

# sorting the  diseases subset
p = 0
date = []
time_of_scan = []
new_name = []
orig_name = []
os.makedirs(f"/media/hdd4/harry/Slides_repo/{args.General_Dx}", exist_ok = True)
os.makedirs(f"/media/hdd4/harry/Slides_repo/{args.General_Dx}/slides", exist_ok = True)

case_IDs= tsv[tsv['General Dx'].isin([args.General_Dx])]['AccessionNumber'].tolist()
def formating(string):
    string = string.replace('\xa0','').strip(' -')
    return string
case_IDs= [formating(x) for x in case_IDs]
print(case_IDs)

    
print('starting')
print(args.move)
### right now the args.move must be False    
assert args.move == False, 'something if off'
safa
### read the metadata
H17 = read_simple_file('/media/hdd4/harry/metadata/H17-20230724.csv')
H18 = read_simple_file('/media/hdd4/harry/metadata/H18-20230724.csv')
H19 = read_simple_file('/media/hdd4/harry/metadata/H19-20230724.csv')
H20 = read_simple_file('/media/hdd4/harry/metadata/H20-20230724.csv')
H21 = read_simple_file('/media/hdd4/harry/metadata/H21-20230724.csv')
H22 = read_simple_file('/media/hdd4/harry/metadata/H22-20230724.csv')
H23 = read_simple_file('/media/hdd4/harry/metadata/H23-20230720.csv')

Heme_dicts = {'H17':H17, 'H18':H18, 'H19':H19, 'H20':H20, 'H21':H21, 'H22':H22, 'H23':H23, }
for _dir in tqdm.tqdm(os.listdir('/pesgisipth/NDPI/')): #/pesgisipth/NDPI/Heme/
    if len(_dir.split(' - '))==2:
        if 'H' in _dir:
            renamed_name = renamefunction(_dir)
            assess_ID = renamed_name.split('_')[0]
            #assess_ID = assess_ID.replace('\xa0','')

            time = assess_ID.split('-')[0]
            if assess_ID in case_IDs: 
                if args.move:
                    assert args.move == False, 'something if off'
                try:
                    subset = Heme_dicts[time][Heme_dicts[time]['specnum_formatted']== assess_ID]
                    
                    assert subset.shape[0]>=1, 'The information could not be find. it might not be a BMA'
                    date.append(_dir.split(' - ')[1].split(' ')[0])
                    time_of_scan.append(_dir.split(' - ')[1].split(' ')[1].split('.ndpi')[0])
                    orig_name.append(_dir)
                    new_name.append(renamed_name)
                    p = p + 1
                    #assert
                    
                    if args.move:
                        shutil.copy('/pesgisipth/NDPI/'+_dir, f'/media/hdd4/harry/Slides_repo/{args.General_Dx}/slides/'+renamed_name)

                except:
                    pass
                    #assert 1==2, 'something off'
data = orig_name
  
# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['orig_name'])
df['new_name'] = new_name
df['date'] = date
df['time_of_scan'] = time_of_scan
  
# print dataframe.
print(df.head())
df.to_csv(f"/media/hdd4/harry/Slides_repo/{args.General_Dx}/table.csv")

print('Find '+str(p)+' cases')
print(f"Saving into {args.General_Dx} diseases")




         


        