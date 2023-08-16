import sys
import os
import pandas as pd
import numpy as np
import tqdm
import shutil
import glob
import argparse

### this script is to remove the wrong ndpi files that doesn't match with the metadata

def read_simple_file(file_dir, columns = ['specnum_formatted','accession_date','part_description', 'barcode']):
    csv = pd.read_csv(file_dir, index_col= 0)
    csv = csv[columns]
    return csv


parser = argparse.ArgumentParser()

parser.add_argument('--General_Dx', type=str,
                    help='Directory in which the cropped patch are stored')

parser.add_argument('--delete', type=bool, default=False,
                    help='If True, delete the wrong ndpi files, also delete the corresponding files in the tsv file')

args = parser.parse_args()


slide_dirs = glob.glob(f'/media/hdd4/harry/Slides_repo/{args.General_Dx}/slides/*.ndpi')

### read the metadata
H17 = read_simple_file('/media/hdd4/harry/metadata/H17-20230724.csv')
H18 = read_simple_file('/media/hdd4/harry/metadata/H18-20230724.csv')
H19 = read_simple_file('/media/hdd4/harry/metadata/H19-20230724.csv')
H20 = read_simple_file('/media/hdd4/harry/metadata/H20-20230724.csv')
H21 = read_simple_file('/media/hdd4/harry/metadata/H21-20230724.csv')
H22 = read_simple_file('/media/hdd4/harry/metadata/H22-20230724.csv')
H23 = read_simple_file('/media/hdd4/harry/metadata/H23-20230720.csv')

Heme_dicts = {'H17':H17, 'H18':H18, 'H19':H19, 'H20':H20, 'H21':H21, 'H22':H22, 'H23':H23, }

for _dir in slide_dirs: #/pesgisipth/NDPI/Heme/
    
    assess_ID = _dir.split('/')[-1].split('_')[0]


    matching_sting = assess_ID+';'+_dir.split('/')[-1].split('_')[1]+\
    ';'+_dir.split('/')[-1].split('_')[2]

    time = assess_ID.split('-')[0]
     
    try:
        subset = Heme_dicts[time][Heme_dicts[time]['specnum_formatted']== assess_ID]
        assert subset.shape[0]>=1, 'The information could not be find. it might not be a BMA'
        reference_lists = subset['barcode'].tolist()
        if matching_sting not in reference_lists:
            assert 1 == 2,''

    except:
        print(f'The case ID is not found. this is not correct, this is the file name {_dir}')
        ### delete the file
        if args.delete:
            os.remove(_dir)
            print(f'File {_dir} is deleted')
        ### accordingly delete the tsv file
        table = pd.read_csv(f"/media/hdd4/harry/Slides_repo/{args.General_Dx}/table.csv", index_col=0)

        table = table[table['new_name']!=_dir.split('/')[-1]]
        table.to_csv(f"/media/hdd4/harry/Slides_repo/{args.General_Dx}/table.csv")
        print(f'File {_dir} is deleted in the table.csv')

        


