'''

Author: @vaibhav.garg
Date: 16th July 2020

'''

import numpy as np
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import tqdm
np.set_printoptions(threshold=sys.maxsize)

def findSeam(I):
	
	# Getting the gradient matrix
	grad = np.abs(cv2.Laplacian(I, cv2.CV_64F))

	# Calculating cumulative grad matrix
	for i in range(1, grad.shape[0]):
		option1 = grad[i-1:i, :]
		option2 = np.append(grad[i-1, 1:], [np.inf]).reshape(1, -1) 
		option3 = np.append([np.inf], grad[i-1, :-1]).reshape(1, -1)
		options = np.concatenate((option1, option2, option3), axis=0)
		
		grad[i, :] = grad[i, :] + np.min(options, axis=0)
	
	# Finding smoothest path
	seam = []
	seam.append(np.argmin(grad[-1, :]))
	for i in range(grad.shape[0]-2, -1, -1):
		if seam[-1] == 0:
			seam.append( np.argmin(grad[i, :2]) )
		elif seam[-1] == grad.shape[1]-1:
			seam.append( np.argmin(grad[i, -2:])-1+seam[-1] )
		else:
			seam.append( np.argmin(grad[i, seam[-1]-1:seam[-1]+2])-1+seam[-1] )

	seam = np.array(seam[::-1]).reshape(-1, 1)
	return seam


def removeSeam(I, seam):
	mask = np.zeros((I.shape[0], I.shape[1]), dtype=np.bool)
	mask[(np.arange(I.shape[0]), seam.flatten())] = True
	newI = I[np.where(mask==False)].reshape(I.shape[0], I.shape[1]-1)
	return newI


def removeMultipleSeams(I, seams):
	mask = np.zeros((I.shape[0], I.shape[1]), dtype=np.bool)
		
	# Removing from all channels at once
	ind1 = np.repeat(np.arange(I.shape[0]), seams.shape[1])
	ind2 = seams.flatten()
	mask[(ind1, ind2)] = True
	
	newI = I[np.where(mask==False)].reshape(I.shape[0], I.shape[1]-seams.shape[1], 3)
	I[mask] = [0, 0, 255]
	return I, newI

def addMultipleSeams(I, seams):
	newI = np.zeros((I.shape[0], I.shape[1]+seams.shape[1], 3), dtype=np.uint8)
	for i in range(I.shape[0]):
		newI[i] = np.insert(I[i], seams[i], I[i, seams[i]], axis=0)
		I[i, seams[i]] = [0, 255, 255]
	
	return I, newI

def runSeamCarving(I, n):

	# Convert to single channel
	try:
		gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	except:
		print("Image given is not RGB")

	# Finding and removing seams
	print("Finding optimal seams")
	seamOffset = np.zeros(gray.shape, dtype=np.int)
	for _ in tqdm.tqdm(range(n)):
		seam = findSeam(gray)
		gray = removeSeam(gray, seam)
		
		try:
			seams = np.concatenate((seams, seam + seamOffset[(np.arange(I.shape[0]), seam.flatten())].reshape(-1, 1) ), axis=1)
		except:
			seams = seam

		for i in range(I.shape[0]):
			seamOffset[i, seam[i, 0]+1:] = seamOffset[i, seam[i, 0]+1:] + 1
		mask = np.zeros(seamOffset.shape, dtype=np.bool)
		mask[(np.arange(I.shape[0]), seam.flatten())] = True		
		seamOffset = seamOffset[np.where(mask==False)].reshape(I.shape[0], seamOffset.shape[1]-1)

	return seams

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("image", help="image path")
parser.add_argument("-W", "--width", help="width change in percentage", type=int, default=100, choices = range(1, 201))
args = parser.parse_args()
print("Arguments given", args)

# Loading image
try:
	I = cv2.imread(args.image)
	print("Loaded image of shape", I.shape)
except:
	print("Error: Unable to load image")

# Core processing
pixel_change = int(abs(100-args.width)*I.shape[1]/100)
print("Change in width of", pixel_change, "pixels")

seams = runSeamCarving(I, pixel_change)

# Expansion or reduction
if args.width > 100:
	orig, carved_img = addMultipleSeams(I, seams)
else:
	orig, carved_img = removeMultipleSeams(I, seams)


# Plotting
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(carved_img, cv2.COLOR_BGR2RGB))
plt.title("Carved")
plt.show()

# Saving
try:
	cv2.imwrite('out.jpg', carved_img)
	print("Image Saved as out.jpg")
except:
	print("Error while saving the image")
