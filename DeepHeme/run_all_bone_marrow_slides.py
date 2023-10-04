import pandas as pd
from DeepHeme_implementation import DeepHeme


def get_bm_wsi_list():
    tsv = pd.read_csv('../wsi_list.csv', sep='\t')
    wsi_name_list = tsv['wsi_name'].tolist()
    wsi_dir_list = tsv['wsi_dir'].tolist()
    return wsi_name_list, wsi_dir_list

# main function
if __name__ == '__main__':
    wsi_name_list, wsi_dir_list = get_bm_wsi_list()
    print(' In total {} slides are identified'.format(len(wsi_name_list)))
    ### create the DeepHeme object
    deepheme = DeepHeme()

    for wsi_name, wsi_dir in zip(wsi_name_list, wsi_dir_list):
        ### get the progress percentage
        progress = wsi_name_list.index(wsi_name)/len(wsi_name_list)
        print('The progress is {}%'.format(progress*100))
        deepheme.set_wsi_dir(wsi_dir=wsi_dir,
                             wsi_name=wsi_name)
        deepheme.run()

        deepheme.deactive()
        break


