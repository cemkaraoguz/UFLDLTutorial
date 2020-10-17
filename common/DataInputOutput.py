''' UFL_DataInputOutput.py
	
	Methods for data input/output
	
	Author: Cem Karaoguz
	Date: 19.02.2015
	Version: 1.0
'''

import os, struct, sys
import numpy as np
import pylab as pl

def fread(fileID, nbits=4, bitpattern='i', skip=0, machinefmt=''):
	'''
	Reads binary data from file and interprets it
	
	Arguments
	fileID		: file object
	nbits		: Number of bits to read
	bitpattern	: pattern of the bits (in struct module format)
	skip		: Number of bytes to skip
	machinefmt	: Order of reading bytes (in struct module format)
	
	Returns
	val			: Array of values read from the file
	'''
	
	if skip>0:
		fileID.seek(skip);
		
	buf = fileID.read(nbits);
	val = struct.unpack(machinefmt+bitpattern, buf);
	
	return np.array(val);

def loadMNISTImages(filename, nData=100000):
	'''
	Returns a 28x28x[number of MNIST images] matrix containing the raw MNIST images
	'''
	
	assert os.path.exists(filename), "ERROR: File not found " + filename
	
	with open(filename, 'rb') as fp:
		
		magic = fread(fp, 4, 'i', 0, '>');
		
		assert magic[0] == 2051, 'Bad magic number in ' + filename
		
		numImages = fread(fp, 4, 'i', 0, '>');
		numRows = fread(fp, 4, 'i', 0, '>');
		numCols = fread(fp, 4, 'i', 0, '>');
		
		numImages = min(nData, numImages);
		
		images = fread(fp, 1*numImages*numRows*numCols, 'B'*numImages*numRows*numCols, 0, '<');
		
		images = np.reshape(images, [numCols, numRows, numImages], order='F');
		images = np.transpose(images, [1, 0, 2]);
		
	fp.close();
	
	
	# Reshape to num pixels x num examples
	images = np.reshape(images, [images.shape[0]*images.shape[1], images.shape[2]]);
	
	# Convert to double and rescale to [0,1]
	#images /= 255.0;

	return images;

def loadMNISTLabels(filename, nData=100000):	
	''' 
	returns a [number of MNIST images]x1 matrix containing the labels for the MNIST images
	'''

	assert os.path.exists(filename), "ERROR: File not found " + filename

	with open(filename, 'rb') as fp:
	
		magic = fread(fp, 4, 'i', 0, '>');
	
		assert magic[0] == 2049, 'Bad magic number in ' + filename
	
		numLabels = fread(fp, 4, 'i', 0, '>');
		
		numLabels = min(nData, numLabels);
		
		labels = fread(fp, 1*numLabels, 'B'*numLabels, 0);

	fp.close();
	
	return labels;

def loadHousingData(filename):
	'''
	Returns a [number of dims]x[number of samples] matrix containing the raw housing data
	'''
	
	assert os.path.exists(filename), "ERROR: File not found " + filename
	
	with open(filename, 'rb') as fp:
		
		numRows = fread(fp, 4, 'i', 0);
		numCols = fread(fp, 4, 'i', 0);
		
		data = fread(fp, 1*numRows*numCols, 'B'*numRows*numCols);
		data = np.reshape(data, [numRows, numCols], order='F');
		
	fp.close();
	
	return data
	
def samplePatches(rawImages, patchWidth, patchHeight, numPatches):
	'''	
	Returns patches of the given size extracted randomly from rawImages
	
	Arguments
	rawImages	: Input images
	patchWidth	: Desired width of patches
	patchHeight	: Desired height of patches
	numPatches	: Total number of patches to sample
	
	Returns
	patches		: List of patches of desired size randomly sampled from given images
	'''
	
	if len(np.shape(rawImages))==3:
		doColor = 0;
		[imWidth, imHeight, numImages] = np.shape(rawImages);
		imColor = 1;
	elif len(np.shape(rawImages))==4:
		doColor = 1;
		[imWidth, imHeight, imColor, numImages] = np.shape(rawImages);
	else:
		print ('ERROR: dimensions of rawImages should be WxHxN or WxHxCxN')
		sys.exit();
	
	# Initialize patches with zeros.  
	patches = np.zeros([patchWidth*patchHeight*imColor, numPatches]);

	# Maximum possible start coordinate
	maxWidth = imWidth - patchWidth;
	maxHeight = imHeight - patchHeight;

	# Sample!
	for num in range(numPatches):
		y = np.random.randint(maxHeight);
		x = np.random.randint(maxWidth);
		img = np.random.randint(numImages);
		
		if doColor:
			p = rawImages[y:y+patchHeight, x:x+patchWidth, :, img];
		else:
			p = rawImages[y:y+patchHeight, x:x+patchWidth, img];
		
		p = np.transpose(p);
		patches[:,num] = np.reshape(p, [patchWidth*patchHeight*imColor]);
		
	return patches;

if __name__ == '__main__':
	
	# Some examples of usage
	
	housing_filename = 'C://develop//python//UFL//data//housing.bin';
	
	data = loadHousingData(housing_filename);
	
	mnist_lbl_filename = 'C://develop//python//UFL//data//train-labels-idx1-ubyte';
	mnist_img_filename = 'C://develop//python//UFL//data//train-images-idx3-ubyte';
	
	labels = loadMNISTLabels(mnist_lbl_filename);	
	images = loadMNISTImages(mnist_img_filename);

	print ("Size of labels: ", np.shape(labels));
	print ("Size of images: ", np.shape(images));
	
	sampleImage = np.reshape(images[:,0], [28, 28]) * 255;
	print (sampleImage)
	pl.imshow(sampleImage, cmap='gray')
	pl.show()
	
	images_small = []
	images_small.append(images[1:5,1:5,123])
	images_small = np.transpose(images_small, [1, 2, 0])
	print (np.shape(images_small))
	
	patches = samplePatches(images_small, 2, 2, 1);
	
	print (images_small)
	print (patches)