# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:24:06 2019

@author: HP
"""
#Importing necessary packages

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

#Module for convolution
def convolve(image,kernel):
    (ih,iw)=image.shape[:2]
    (kh,kw)=kernel.shape[:2]    # taking the spatial dimensions of the image, along with
       # the spatial dimensions of the kernel
       # "pad" the borders of the input image
    pad =(kw - 1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output=np.zeros((ih, iw),dtype="float32")
    
    #Applying convolution to the given image
    for y in np.arange(pad,ih+pad):
        for x in np.arange(pad,iw+pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
        
    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# constructing average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
#larger the kernel,more blurred the image will be
# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")
# construct the Laplacian kernel used to detect edge-like regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((#detetcting vertical changes in the gradient of the image
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],#detecting horizontal changes in the gradient of the image
	[0, 0, 0],
	[1, 2, 1]), dtype="int")
#Now we construct the kernel bank
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY)
)
# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, kernel) in kernelBank:
	# apply the kernel to the grayscale image using both
	print("[INFO] applying {} kernel".format(kernelName))
	convoleOutput = convolve(gray, kernel)
	opencvOutput = cv2.filter2D(gray, -1, kernel)

	# show the output images
	cv2.imshow("original", gray)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()