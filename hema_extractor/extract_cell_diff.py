### this script will use rule based method to extract the cell type proportion
### from the reports


import os
import glob
import sys
import pandas as pd
from tqdm import tqdm
import pandas as pd  
import regex
    
# Function to extract text from RTF content  
def rtf_to_plain_text(rtf_content):  
    text_pattern = regex.compile(r'\{\\[a-zA-Z]+\}|\\[a-zA-Z]+|\s?\\[a-zA-Z]+|\{|\}', regex.MULTILINE)  
    return text_pattern.sub('', rtf_content) 

tsv = pd.read_csv('../text_data/Heme_files/H17-20230724.csv')
print(tsv['text_data_final'].iloc[100])
'''
print(tsv['text_data_final'].iloc[100])

# Replace the RTF column with plain text  
rtf_column = 'text_data_final'  
tsv['plain_text_final'] = tsv['text_data_final'].apply(rtf_to_plain_text) 
print('------------')
### overwrite the original tsv file
tsv.to_csv('../text_data/Heme_files/H18-20230724.csv', index=False)
'''

print('------------')
print(tsv['plain_text_final'].iloc[100])