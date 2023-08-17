import cv2
import numpy as np
import matplotlib.pyplot as plt
### convert 4 channels Image object to 3 channels
def convert_4_to_3(img):
    ### 4 to 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ### convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

### conver image to grayscale
def convert_to_grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

import cv2
import numpy as np
def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

### calculate the Fourier transform of an image
### also have a function to visualize the Fourier transform
def fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum