

#importing the necessary packages
import numpy as np
import argparse
import cv2

# constructing the argument parse and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# loading the image
image = cv2.imread(args["image"])

# defining the list of boundaries
boundaries = [
	([17,15,100], [50, 56, 200]),#for red
	([86, 31, 4], [220, 88, 50]),#for blue
	([25,146,190], [62, 174,250]),#for yellow
	([103, 86, 65], [145, 133, 128])# for gray
]

# looping over the boundaries
for (lower, upper) in boundaries:
	# creating NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)#a binary mask will be returned where white pixels determine the pixels that fall in the range and black represent not
	output = cv2.bitwise_and(image, image, mask = mask)#show pixels in the image that correspond to the white pixels in the mask

	# show the images
	cv2.imshow("images", np.hstack([image, output]))#creating a horizontal stack of images
	cv2.waitKey(0)