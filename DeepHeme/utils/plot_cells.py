'''
This function is used to visulize the images in a folder. 
The input is a folder path, and the output is a plot of images.
'''
import numpy as np


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

class ploting_multiple_images():
    def __init__(self, folder_path, number_of_images = 16, random = False):
        self.folder_path = folder_path
        self.number_of_images = number_of_images
        self.random = random

    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import os

        # Get the list of all files in directory tree at given path
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.folder_path):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        if self.random:
            import random
            selected_listOfFiles = random.sample(listOfFiles, self.number_of_images)
        else:
            selected_listOfFiles = listOfFiles[:self.number_of_images]

            

        # Plot the images in the folder as a square matrix
        fig = plt.figure(figsize=(10,10))
        # check if the number of images is a square number
        # if not, add the extra space to the last row
        # if yes, plot the images as a square matrix
        # if the number of images is less than 4, plot them in a row
        if self.number_of_images < 4:
            for i in range(self.number_of_images):
                img = mpimg.imread(selected_listOfFiles[i])
                fig.add_subplot(1, self.number_of_images, i+1)
                plt.imshow(img)
                plt.axis('off')
        elif is_square(self.number_of_images):
            for i in range(self.number_of_images):
                img = mpimg.imread(selected_listOfFiles[i])
                fig.add_subplot(int(np.sqrt(self.number_of_images)),
                                 int(np.sqrt(self.number_of_images)), i+1)
                plt.imshow(img)
                plt.axis('off')
        else:
            assert is_square(self.number_of_images) == False, "The number of images should not be a square number"
            for i in range(self.number_of_images):
                img = mpimg.imread(selected_listOfFiles[i])
                fig.add_subplot(int(np.sqrt(self.number_of_images)), int(np.sqrt(self.number_of_images)+1), i+1)
                plt.imshow(img)
                plt.axis('off')
                
def crop_image(image_dir, output_dir):
    import os
    import cv2
    import numpy as np
    '''
    :param image_dir: the dir of the images
    :param output_dir: the dir of the output images'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        image_name = image.split('.')[0]
        image_output_path = os.path.join(output_dir, image_name + '.png')
        img = cv2.imread(image_path)
        img = img[16:80, 16:80]
        cv2.imwrite(image_output_path, img)


         


        