import os
import cv2
import numpy as np
import openslide
import matplotlib.pyplot as plt
from tqdm import tqdm
def visualize_whole_slide_using_open_slide(image_dir = None,
                         level = 0, return_image = False, show = True):
    ### Visualize whole slide image
    '''
    image_dir: path to whole slide image
    level: level of whole slide image
    return_image: whether to return the image
    show: whether to show the image
    '''
    img_path = image_dir
    slide = openslide.OpenSlide(img_path)
    region = (0, 0)
    #size = (6600, 3240)
    size = slide.level_dimensions[level]
    region = slide.read_region(region, level, size)
    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(region)
    
    if return_image:
        return region


def visualize_whole_slide_using_patches(patch_dir = None, resize_patch = False, resized_patch_size = (10,10)):
    ### Visualize whole slide image by stacking a collection of patches
    ### patches name are in the format of "patch_x_y.png， where x and y are the coordinates of the patch“
    '''
    patch_dir: path to patches
    resize_patch: whether to resize the patches
    resized_patch_size: size of resized patches
    '''
    patch_list = os.listdir(patch_dir)
    patch_list.sort()
    # x and y should be divided by 512 because the size of each patch is 512*512
    x = [int(int(patch.split('_')[1])/512) for patch in patch_list]
    y = [int(int(patch.split('_')[2].split('.')[0])/512) for patch in patch_list]
    # get the maximum x and y
    max_x = max(x)
    max_y = max(y)
    # create a matrix to store the patches
    patch_matrix = np.zeros((max_x+1, max_y+1, resized_patch_size[0], resized_patch_size[1], 3))
    # read patches and store them in the matrix

    for patch in tqdm(patch_list):
        ### read patch
        patch_path = os.path.join(patch_dir, patch)
        patch_img = cv2.imread(patch_path)
        ### resize patch
        if resize_patch:
            patch_img = cv2.resize(patch_img, resized_patch_size)
        ### get x and y coordinates
        
        ### store patch in the matrix
        patch_matrix[x, y, :, :, :] = patch_img.resize(resized_patch_size[0], resized_patch_size[1], 3)
    ### stack patches
    patch_stack = np.vstack([np.hstack([patch_matrix[i, j, :, :, :] for j in range(patch_matrix.shape[1])]) for i in range(patch_matrix.shape[0])])
    ### visualize
    return patch_stack


class WholeSlideImageVisualizer: 
    '''
    #Visualize whole slide image by stacking a collection of patches
    '''
    import cv2  
    import os  
    import numpy as np  
    from concurrent.futures import ThreadPoolExecutor
      
    from tqdm import tqdm
    def __init__(self):
        pass 
    
    def _stacking_image_collection(self, input_folder, resize_patch = False, resized_patch_size = (10,10)):  
        # Get list of all patch files in the folder  
        patch_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]  
    
        # Read and store patches in parallel using ThreadPoolExecutor  
        with ThreadPoolExecutor() as executor:  
            patches = list(executor.map(self._read_patch, [os.path.join(input_folder, f) for f in patch_files]))  
    
        # Extract x, y coordinates from file names and store them as tuples (x, y, patch)  
        coords_patches = [int((int(f.split("_")[1])/512), int(int(f.split("_")[2].split(".")[0])/512), patch) for f, patch in zip(patch_files, patches)]  
    
        # Find the maximum x and y coordinates to determine the size of the final image  
        max_x = max(coords_patches, key=lambda x: x[0])[0]  
        max_y = max(coords_patches, key=lambda x: x[1])[1] 
        print(max_x,max_y)
    
        # Create an empty canvas for the final image  
        patch_shape = resized_patch_size
        final_image = np.zeros((patch_shape[0] * (max_x + 1), patch_shape[1] * (max_y + 1), 3), dtype=np.uint8)  
    
        # Place each patch at its corresponding position on the canvas  
        for x, y, patch in tqdm(coords_patches):  
            if resize_patch:
                patch = cv2.resize(patch, patch_shape)
            final_image[x * patch_shape[0]:(x + 1) * patch_shape[0], y * patch_shape[1]:(y + 1) * patch_shape[1]] = patch  
    
        return final_image 
    


