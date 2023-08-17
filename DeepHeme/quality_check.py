### This file contains functions that check the quality of the image. ###
### It have all the functions that help remove slides that is not bone marrow aspirate ###
### It also have functions that help remove slides that is not stained properly ###

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils.image_basics import convert_4_to_3, convert_to_grayscale, laplacian, fourier

def last_min_before_last_max(local_minima, local_maxima, last_n=1):
    """Returns the last local minimum before the last local maximum"""
    
    for i in range(len(local_minima)-1, -1, -1):
        if local_minima[i] < local_maxima[-last_n]:
            return local_minima[i]


def get_white_mask(image, verbose=False):
    """ Return a mask covering the whitest region of the image. """

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a histogram of the pixel intensities of the image
    bins = 64
    histogram = cv2.calcHist([gray_image], [0], None, [bins], [0, 256])

    # Calculate the bin midpoints
    bin_midpoints = np.linspace(0, 256, bins+1)[1:] - (256/bins/2)

    # Smooth out the histogram to remove small ups and downs but keep the large peaks
    histogram = cv2.GaussianBlur(histogram, (5, 5), 0)

    if verbose:
        # Display the histogram
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(bin_midpoints, histogram)
        plt.xlim([0, 256])
        plt.show()

    # There are multiple peaks in the histogram
    # The first peak is the covering label and we want to find a mask that covers the covering label
    # The last peak is the background and we want to find a mask that covers the background

    # grab a list of local minima positions and a list of local maxima positions
    # the first local minima is the first peak, which is the covering label
    # the last local maxima is the last peak, which is the background

    # find the local minima
    local_minima = []
    for i in range(1, len(histogram)-1):

        if histogram[i-1] > histogram[i] < histogram[i+1]:
            local_minima.append(bin_midpoints[i])

    # find the local maxima
    local_maxima = []
    for i in range(0, len(histogram)-1):

        # if the index is the left most boundary, then no need to compare the left value
        if i == 0:
            if histogram[i] > histogram[i+1]:
                local_maxima.append(bin_midpoints[i])
        elif histogram[i-1] < histogram[i] > histogram[i+1]:
            local_maxima.append(bin_midpoints[i])

    if verbose:
        # plot the local minimum and maximum positions, minima are blue, maxima are red
        # plot the minimal maxima positions as vertical lines
        # make the line corresponding to first_min_after_first_max(local_minima, local_maxima) longer than the rest
        plt.figure()
        plt.title("Local Minima and Maxima")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(bin_midpoints, histogram)
        plt.xlim([0, 256])
        plt.vlines(last_min_before_last_max(
            local_minima, local_maxima), 0, max(histogram), colors="g")
        plt.vlines(local_minima, 0, 1000, colors="b")
        plt.vlines(local_maxima, 0, 3000, colors="r")
        plt.show()

    # get a mask that contains all pixels with intensity smaller than the first local minimum right after the first peak
    mask = np.zeros(gray_image.shape, dtype="uint8")
    mask[gray_image > last_min_before_last_max(
        local_minima, local_maxima)] = 255

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    return mask


def get_background_mask(image, erosion_radius=35, median_blur_size=35, verbose=False):
    """ Returns a mask that covers the complement of the obstructor in the image. """

    mask = get_white_mask(image, verbose=verbose)

    # Now we use the mask to mask the image.
    # Start by applying a median blur to the mask to get rid of salt and pepper noises
    mask = cv2.medianBlur(mask, median_blur_size)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Median Blurred Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    # Then invert the mask
    mask = cv2.bitwise_not(mask)

    # Then thin the mask to get rid of the obstructor
    kernel = np.ones((erosion_radius, erosion_radius), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Eroded Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    # Remove all connected components in the black region of the mask that are smaller than 15000 pixels
    # This removes small holes in the mask

    # invert the mask
    mask = cv2.bitwise_not(mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 15000:
            mask[labels == i] = 0

    # invert the mask again
    mask = cv2.bitwise_not(mask)

    if verbose:
        # Display each connected component in the mask
        plt.figure()
        plt.title("Connected Components")
        plt.imshow(labels)
        plt.show()

        # display the mask
        plt.figure()
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    return mask


def get_threshold(img, prop_black=0.3, bins=1024):
    # tally up the histogram of the image's greyscale values
    histogram = np.zeros(bins)
    for pixel in img:
        histogram[pixel] += 1
    # choose the most generous threshold that gives a proportion of black pixels greater than prop_black
    threshold = 0
    for i in range(bins):
        if np.sum(histogram[:i]) / np.sum(histogram) >= prop_black:
            threshold = i
            break
    return threshold


""" Apply a number of cv2 filters to the image specified in image_path. If verbose is True, the image will be displayed after each filter is applied, and the user will be prompted to continue. If verbose is False, the image will not be displayed, and the user will not be prompted to continue. """


def get_high_single_channel_signal_mask(image, prop_black=0.75, bins=1024, median_blur_size=3, dilation_kernel_size=9, verbose=False, channel =None):
    """
    Return a mask that covers the high blue signal in the image.
    """

    # # Apply pyramid mean shift filtering to image
    # image = cv2.pyrMeanShiftFiltering(image, 21, 51)
    # if verbose:
    #     try:
    #         # display the image and pause the execution until the user presses a key
    #         cv2.imshow("PMSF", image)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # # Apply a blur filter to image
    # image = cv2.blur(image, (5,5))
    # if verbose:
    #     try:
    #         # display the image and pause the execution until the user presses a key
    #         cv2.imshow("Blur", image)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # Convert to float32 to prevent overflow during division
    image = image.astype(np.float32)

    # Compute the sum over the color channels
    sum_channels = np.sum(image, axis=2, keepdims=True)

    # To avoid division by zero, we can add a small constant
    sum_channels = sum_channels + 1e-7

    if channel is None:
        raise ValueError("Please specify the channel to use")
    elif channel not in ['r', 'g', 'b']:
        raise ValueError("Please specify the channel to use as 'r', 'g', or 'b'")
    elif channel == 'b':
        # Normalize the blue channel by the sum
        image[:, :, 0] = image[:, :, 0] / sum_channels[:, :, 0] 
        # Now, image has the normalized blue channel, and all other channels as they were.
        # If you want to zero out the other channels, you can do it now
        image[:, :, 1] = 0  # zero out green channel
        image[:, :, 2] = 0  # zero out red channel
    elif channel == 'g':
        # Normalize the green channel by the sum
        image[:, :, 1] = image[:, :, 1] / sum_channels[:, :, 0] 
        # Now, image has the normalized green channel, and all other channels as they were.
        # If you want to zero out the other channels, you can do it now
        image[:, :, 0] = 0
        image[:, :, 2] = 0
    elif channel == 'r':
        # Normalize the red channel by the sum
        image[:, :, 2] = image[:, :, 2] / sum_channels[:, :, 0] 
        # Now, image has the normalized red channel, and all other channels as they were.
        # If you want to zero out the other channels, you can do it now
        image[:, :, 0] = 0
        image[:, :, 1] = 0
    else:
        ### stop the implementation because the channel is not specified correctly
        raise ValueError("Please specify the channel to use as 'r', 'g', or 'b'")

    # Before saving, convert back to uint8
    image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8)

    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow(f"{channel} Channel", image)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("Grayscale", image)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # # Apply a laplacian filter to image
    # image = cv2.Laplacian(image, cv2.CV_64F)
    # image = np.absolute(image)  # Absolute value
    # image = np.uint8(255 * (image / np.max(image)))  # Normalize to 0-255
    # if verbose:
    #     try:
    #         # display the image and pause the execution until the user presses a key
    #         cv2.imshow("Laplacian", image)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # apply a median blur to the image to get rid of salt and pepper noise
    image = cv2.medianBlur(image, median_blur_size)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("Median Blur", image)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # dilate the image to get rid of small holes
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("Dilate", image)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # # erode the image to get rid of small protrusions
    # kernel = np.ones((5,5),np.uint8)
    # image = cv2.erode(image, kernel, iterations = 1)
    # if verbose:
    #     try:
    #         # display the image and pause the execution until the user presses a key
    #         cv2.imshow("Erode", image)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # threshold the image to get a black and white image with solid white areas, be generous with the threshold
    # tally up the histogram of the image's greyscale values, and choose the threshold just before the peak
    threshold = get_threshold(image, prop_black=prop_black, bins=bins)
    image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("Threshold", image)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    return image


def first_min_after_first_max(local_minima, local_maxima, first_n=2):
    """Returns the first local minimum after the first local maximum"""
    for i in range(len(local_minima)):
        if local_minima[i] > local_maxima[first_n-1]:
            return local_minima[i]


def get_obstructor_mask(image, erosion_radius=25, median_blur_size=25, verbose=False, first_n=2, apply_blur=False):
    """ Returns a mask that covers the complement of the obstructor in the image. """

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a histogram of the pixel intensities of the image
    bins = 64
    histogram = cv2.calcHist([gray_image], [0], None, [bins], [0, 256])

    # Calculate the bin midpoints
    bin_midpoints = np.linspace(0, 256, bins+1)[1:] - (256/bins/2)

    if apply_blur:
        # Apply a Gaussian blur to the function
        histogram = cv2.GaussianBlur(histogram, (5, 5), 0)

    if verbose:
        # Display the histogram
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(bin_midpoints, histogram)
        plt.xlim([0, 256])
        plt.show()

    # There are multiple peaks in the histogram
    # The first peak is the covering label and we want to find a mask that covers the covering label
    # The last peak is the background and we want to find a mask that covers the background

    # grab a list of local minima positions and a list of local maxima positions
    # the first local minima is the first peak, which is the covering label
    # the last local maxima is the last peak, which is the background

    # find the local minima
    local_minima = []
    for i in range(1, len(histogram)-1):

        if histogram[i-1] > histogram[i] < histogram[i+1]:
            local_minima.append(bin_midpoints[i])

    # find the local maxima
    local_maxima = []
    for i in range(0, len(histogram)-1):

        # if the index is the left most boundary, then no need to compare the left value
        if i == 0:
            if histogram[i] > histogram[i+1]:
                local_maxima.append(bin_midpoints[i])
        elif histogram[i-1] < histogram[i] > histogram[i+1]:
            local_maxima.append(bin_midpoints[i])

    if verbose:
        # plot the local minimum and maximum positions, minima are blue, maxima are red
        # plot the minimal maxima positions as vertical lines
        # make the line corresponding to first_min_after_first_max(local_minima, local_maxima) longer than the rest
        plt.figure()
        plt.title("Local Minima and Maxima")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(bin_midpoints, histogram)
        plt.xlim([0, 256])
        plt.vlines(first_min_after_first_max(local_minima, local_maxima,
                   first_n=first_n), 0, max(histogram), colors="g")
        plt.vlines(local_minima, 0, 1000, colors="b")
        plt.vlines(local_maxima, 0, 3000, colors="r")
        plt.show()

    # get a mask that contains all pixels with intensity smaller than the first local minimum right after the first peak
    mask = np.zeros(gray_image.shape, dtype="uint8")
    mask[gray_image < first_min_after_first_max(
        local_minima, local_maxima, first_n=first_n)] = 255

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    # Now we use the mask to mask the image.
    # Start by applying a median blur to the mask to get rid of salt and pepper noises
    mask = cv2.medianBlur(mask, median_blur_size)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Median Blurred Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    # Then invert the mask
    mask = cv2.bitwise_not(mask)

    # Then thin the mask to get rid of the obstructor
    kernel = np.ones((erosion_radius, erosion_radius), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Eroded Mask")
        plt.imshow(mask, cmap="gray")
        plt.show()

    return mask


def marrow_boxing(mask, image, background_mask, box_ratio=0.1, output_dir=None, verbose=False):
    """ Put boxes based on mask from mask_path on image from image_path. If output_path is not None, save the image to output_path. 
    Else, save to working directory. If verbose is True, display the image. 
    The mask from background path is to remove the background from the marrow mask. """

    # create a new mask, this new mask is constructed by added a box around each pixel in the original mask
    # the box is 2*box_radius+1 pixels wide centered at each pixel
    # the radius is a center proprotion of the minimum of the image width and height

    box_radius = int(box_ratio * min(image.shape[:2]))

    # create a new mask
    new_mask = np.zeros_like(mask, dtype="uint8")

    # get the coordinates of all the white pixels in the mask
    white_pixels = np.where(mask == 255)

    # for each white pixel, add a box around it
    for i in range(len(white_pixels[0])):
        # get the coordinates of the current white pixel
        row = white_pixels[0][i]
        col = white_pixels[1][i]

        # add a box around the current white pixel, if the box is out of bounds, crop the part that is out of bounds
        # the box is 2*box_radius+1 pixels wide centered at the current white pixel
        # the radius is a center proprotion of the minimum of the image width and height
        # the box is added to the new mask
        new_mask[max(0, row-box_radius):min(new_mask.shape[0], row+box_radius+1),
                 max(0, col-box_radius):min(new_mask.shape[1], col+box_radius+1)] = 255

    if verbose:
        # display the original mask and the new mask side by side
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Original Mask")
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("New Mask")
        plt.imshow(new_mask, cmap="gray")
        plt.show()

    # now display the original image, the mask, and then the new_mask layed on top of the original image in color green, with transparency 0.5
    # open the image using OpenCV

    # convert the image to RGB format for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get the coordinates of all the black pixels in the background mask
    background_black_pixels = np.where(background_mask == 0)

    # for each white pixel, set the corresponding pixel in the new mask to 0
    for i in range(len(background_black_pixels[0])):
        # get the coordinates of the current white pixel
        row = background_black_pixels[0][i]
        col = background_black_pixels[1][i]

        # set the corresponding pixel in the new mask to 0
        new_mask[row, col] = 0

    # display the original image, the mask, and the new_mask layed on top of the original image in color green, with transparency 0.3
    # the original image should have transparency 1.0, and the mask should have transparency 0.3
    # the original and the mask should be displayed side by side
    # the new_mask should be displayed below the other two

    # convert the single channel mask to a 3-channel mask
    new_mask_colored = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)

    # change the color of the mask to green
    new_mask_colored[:, :, 0] = 0  # Zero out the blue channel
    new_mask_colored[:, :, 2] = 0  # Zero out the red channel

    # now display the original image, the mask, and then the new_mask_colored layed on top of the original image in color green, with transparency 0.5
    # Make sure the mask is put on the original image in the correct order
    overlayed_image = cv2.addWeighted(image, 1.0, new_mask_colored, 0.2, 0.0)

    if verbose:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("New Mask on Original Image")
        plt.imshow(overlayed_image)
        plt.show()

    # Save the new mask on the original image

    # Convert the overlayed image to BGR format for OpenCV
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

    if output_dir is None:
        # save to working directory
        cv2.imwrite("marrow_boxing.png", overlayed_image)
    else:
        # save to output_path
        output_path = os.path.join(output_dir, "marrow_boxing.png")
        cv2.imwrite(output_path, overlayed_image)

    return new_mask


def get_top_view_preselection_mask(image, verbose=False, RGB = False):
    """ The input is a cv2 image which is a np array in BGR format. Output a binary mask used to region preselection. """
    if RGB:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    high_blue = get_high_single_channel_signal_mask(
        image, prop_black=0.75, median_blur_size=3, dilation_kernel_size=9, verbose=verbose, channel='b')

    # get the obstructor mask
    obstructor_mask = get_obstructor_mask(image, verbose=verbose)

    # get the background mask
    background_mask = get_background_mask(image, verbose=verbose)

    # combine the two masks
    final_blue_mask = cv2.bitwise_and(high_blue, obstructor_mask)

    final_mask = marrow_boxing(final_blue_mask, image,
                               background_mask, box_ratio=0.12, verbose=verbose)

    return final_mask


def is_touch_prep(image, verbose=False, erosion_radius=75, median_blur_size=75):
    """ Return whether or not the image (top view of a whole slide image) is a touch prep. """

    background_mask = get_background_mask(
        image, verbose=verbose, erosion_radius=erosion_radius, median_blur_size=median_blur_size)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Background Mask")
        plt.imshow(background_mask, cmap="gray")
        plt.show()

    # if the white pixels in background mask  are less than 25% of the total pixels, then the image is a touch prep
    non_removed_prop = np.sum(background_mask) / \
        (np.prod(background_mask.shape) * 255)

    return non_removed_prop < 0.25


def is_peripheral_blood(image, verbose = False, erosion_radius=75, median_blur_size=75):
    """ Return whether or not the image (top view of a whole slide image) is a peripheral blood smear. """
    if verbose:
        print('Please make sure the image is in BGR format')
    background_mask = get_background_mask(
        image, verbose=verbose, erosion_radius=erosion_radius, median_blur_size=median_blur_size)

    if verbose:
        # display the mask
        plt.figure()
        plt.title("Background Mask")
        plt.imshow(background_mask, cmap="gray")
        plt.show()

    # if the white pixels in background mask  are less than 25% of the total pixels, then the image is a touch prep
    non_removed_prop = np.sum(background_mask) / \
        (np.prod(background_mask.shape) * 255)

    if non_removed_prop<0.4:
        return False
    else:
        # calculating the laplacian of the image
        laplacian_image_var = laplacian(image)
        raise NotImplementedError, print("Not calculated yet")

        if laplacian_image_var > 500:
            return True
        else:
            return False
        
def is_iron_staining(image, verbose = False, erosion_radius=75, median_blur_size=75):
    """ The input is a cv2 image which is a np array in BGR format. Output a binary mask used to region preselection. """
    if verbose:
        print('Please make sure the image is in BGR format')
    background_mask = get_background_mask(
        image, verbose=verbose, erosion_radius=erosion_radius, median_blur_size=median_blur_size)
    
    if verbose:
        # display the mask
        plt.figure()
        plt.title("Background Mask")
        plt.imshow(background_mask, cmap="gray")
        plt.show()

    # if the white pixels in background mask  are less than 25% of the total pixels, then the image is a touch prep
    non_removed_prop = np.sum(background_mask) / \
        (np.prod(background_mask.shape) * 255)

    
    
    
    high_red = get_high_single_channel_signal_mask(
        image, prop_black=0.75, median_blur_size=3, dilation_kernel_size=9, verbose=verbose, channel='r')

    

    if non_removed_prop<0.4:
        return False
    else:
        # calculating the laplacian of the image
        laplacian_image_var = laplacian(image)
        raise NotImplementedError, print("Not calculated yet")
    
        non_removed_prop = np.sum(background_mask) / \
        (np.prod(background_mask.shape) * 255)
        return non_removed_prop > 0.5
    
    # please this onhold, I might need to calculate the laplacian as well
    

    

if __name__ == '__main__':
    image_rel_path = "bone_marrow_example.png"
    image = cv2.imread(image_rel_path)

    print('Is touch prep:', is_touch_prep(image, verbose=True))

    # get the top view preselection mask
    mask = get_top_view_preselection_mask(image, verbose=True)

    # display the mask
    plt.figure()
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.show()

    touch_prep_path = "touch_prep_example.png"

    image = cv2.imread(touch_prep_path)
    print("Is touch prep:", is_touch_prep(image, verbose=True))
