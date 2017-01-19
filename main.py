from sklearn import datasets
import numpy as np
import csv
import cv2

# for loop iterates through folder containing the images
# saves every value for each image in list (?)
# each list equals one row in the final csv output file
# output csv file

# Load Input Image
inputImage = cv2.imread('/Users/patrickmohr/Code/Python/imageDataExtractor/resources/test2.jpg', 1)
gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
imageHeight, imageWidth, numberOfColorChannels = inputImage.shape

# Image Height & Width --------------
imageHeight, imageWidth = inputImage.shape[:2]
print "y:", imageHeight, "x:", imageWidth

# Resolution ------------------------
imageResolution = imageHeight * imageWidth
print "Resolution:", imageResolution

# File Size -------------------------

# Bit Depth -------------------------
bitDepth = inputImage.dtype
if bitDepth == 'uint8':
    bitDepth = 8 * numberOfColorChannels
elif bitDepth == 'uint16':
    bitDepth = 16 * numberOfColorChannels
print "Bit depth:", bitDepth

# Feature Detection
sift = cv2.SIFT()
keyPoints = sift.detect(inputImage, None)
numberOfKeypoints = len(keyPoints)
print "Keypoints:", numberOfKeypoints
# sift evtl. als teil der grun

# Compression -----------------------

# Write CSV-File --------------------
RESULTS = [[imageHeight, imageWidth, imageResolution, bitDepth, numberOfKeypoints]]
with open("imageData.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(RESULTS)