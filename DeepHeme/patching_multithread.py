import os
import multiprocessing
import pandas as pd
import argparse
def get_image_files(latest = False, Diagnose = None):
    '''
    latest: a boolean value, if true, only the latest image will be processed
    Diagnose: a string representing the diagnosis of the slides
    '''

    
    # Replace 'your_folder_path' with the actual path to the folder containing the images
    assert Diagnose != None, "Please specify the diagnosis of the slides"
    folder_path = f"/media/hdd4/harry/Slides_repo/{Diagnose}/slides"
    
    
    if latest:
        image_files = []
        table = pd.read_csv(f"/media/hdd4/harry/Slides_repo/{Diagnose}/table.csv")
        table['caseID'] = table['new_name'].apply(lambda x: x.split('_')[0]+'_'+x.split('_')[1]+'_'+x.split('_')[2])
        lists = []
        for _group in table.groupby('caseID'):
            image_files.append(_group[1].sort_values(['date', 'time_of_scan'], ascending=False).iloc[0]['new_name'])
    else:
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.ndpi') ]
    patch_path = folder_path.replace('slides','patches')

    ### create the patch folder if it does not exist
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)
    return [(patch_path, folder_path, image_file) for image_file in image_files]

def image_patching_function(save_destination, image_dic, image_dir):
    os.system(f"python Patching.py --save_destination {save_destination} --wsi_dir {image_dic} --wsi_name {image_dir}" )



def main(args):
    # Get the list of image files
    image_files = get_image_files(latest = True, Diagnose = args.Diagnosis)
    print(f"{len(image_files)} files are identified")

    # Number of processes to create (you can adjust this based on the number of CPU cores you want to utilize)
    max_num_processes = multiprocessing.cpu_count()
    num_processes = args.number_of_threads # I could use 64 of them
    ### assert num_processes < max_num_processes
    assert num_processes < max_num_processes, "The number of threads should be less than the number of cores"

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Define the output folder for processed images (replace 'output_folder' with your desired folder name)
    

    # Use pool.starmap() to run the process_image function in parallel
    pool.starmap(image_patching_function, [(image_dir[0], image_dir[1], image_dir[2]) for (image_dir) in image_files])

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print("Image processing complete.")
if __name__ == '__main__':
    # Define the command line arguments
    argparser = argparse.ArgumentParser(description='WSI patching with multithreading')
    argparser.add_argument('--Diagnosis', default=None, type=str, help='Diagnosis of the slides')
    argparser.add_argument('--number_of_threads', default=16, type=int, help='Number of threads to use')
    args = argparser.parse_args()
    main(args)
