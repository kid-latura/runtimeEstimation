from sklearn import datasets
import numpy
from os import listdir
from os.path import isfile, join
import csv
import cv2

# for loop iterates through folder containing the images
# saves every value for each image in list (?)
# each list equals one row in the final csv output file
# output csv file

dataCSV = open('imageData.csv', 'w')
writer = csv.writer(dataCSV, dialect='excel')

# Load Input Images
imageFolder='/Users/patrickmohr/Code/Python/runtimeEstimation/resources/images/'
onlyFiles = [f for f in listdir(imageFolder) if isfile(join(imageFolder, f))]
inputImages = numpy.empty(len(onlyFiles), dtype=object)
#inputImagesGrey = numpy.empty(len(onlyFiles), dtype=object)

for n in range(0, len(onlyFiles)):
  inputImages[n] = cv2.imread(join(imageFolder, onlyFiles[n]), 1)
  #inputImagesGrey[n] = cv2.cvtColor(join(imageFolder, onlyFiles[n]), cv2.COLOR_BGR2GRAY)

# Extract Params
for n in range(0, len(inputImages)):
    imageHeight, imageWidth, numberOfColorChannels = inputImages[n].shape

    resolution = imageHeight * imageWidth

    bitDepth = inputImages[n].dtype
    if bitDepth == 'uint8':
        bitDepth = 8 * numberOfColorChannels
    elif bitDepth == 'uint16':
        bitDepth = 16 * numberOfColorChannels

    sift = cv2.SIFT() # sift evtl. als teil der grun
    keyPoints = sift.detect(inputImages[n], None)
    numberOfKeypoints = len(keyPoints)

    RESULTS = ([imageHeight, imageWidth, resolution, bitDepth, numberOfKeypoints])
    writer.writerow(RESULTS)


