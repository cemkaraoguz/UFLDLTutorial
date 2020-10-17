''' SelfTaughtLearning.py
	
	Implementation of a network to achieve handwritten digit recognition.
	The network has the following topology:
	
	Input -> SoftICA -> Softmax -> Output
	
	Input	: Handwritten digit images from MNIST dataset
	SoftICA	: achieves feature extraction
	Softmax	: achieves classification
	
	Author: Cem Karaoguz
	Date: 27.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
import scipy.signal

from UFL.common import DataInputOutput, DataNormalization, Visualization
from UFL.PCA import PCA
from UFL.SoftICA import SoftICA
from UFL.Softmax import Softmax

def convolveAndPool(images, W, poolDim):
	''' Returns the convolution of the features given by W with
	the given images. 
	
	Arguments
	images		: large images to convolve with, matrix in the form 
	              images(r, c, image number)
	W			: filterbank, is of shape (filterDim,filterDim,numFilters)
	poolDim		: dimension of square pooling
	
	Returns
	features	: matrix of convolved and pooled features in the form
	              features(imageRow, imageCol, featureNum, imageNum)
	'''
	imageDimX = np.shape(images)[0];
	imageDimY = np.shape(images)[1];
	numImages = np.shape(images)[2];
	filterDimX = np.shape(W)[0];
	filterDimY = np.shape(W)[1];
	numFilters = np.shape(W)[2];
	convDimX = imageDimX - filterDimX + 1;
	convDimY = imageDimY - filterDimY + 1;
	features = np.zeros([convDimX/poolDim, convDimY/poolDim, numFilters, numImages]);

	poolMat = np.ones([poolDim]);

	for imageNum in range(numImages):
		for filterNum in range(numFilters):
		
			filter = W[:,:,filterNum];
			
			# Flip the feature matrix because of the definition of convolution
			filter = np.rot90(filter, 2);
			
			# Obtain the image
			im = images[:, :, imageNum];
			
			resp = scipy.signal.convolve2d(im, filter, mode='valid');
			
			# Apply pooling on "resp" to get the hidden activation "act"
			if 0:
				# Mean pooling
				poolingFilter = np.ones([poolDim, poolDim]) * (poolDim * poolDim)**(-1);
				act = scipy.signal.convolve2d(resp, poolingFilter, mode='valid');
			else:
				# Square root pooling
				poolingFilter = np.ones([poolDim, poolDim]);
				aux1 = resp**2;
				act = np.sqrt(scipy.signal.convolve2d(aux1, poolingFilter, 'valid'));
				
			features[:, :, filterNum, imageNum] = act[0:convDimX-poolDim+1:poolDim, 0:convDimY-poolDim+1:poolDim];
			
	return features

if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Learning orthagonal bases of images of handwritten digits (MNIST dataset)
	# --------------------------

	mnist_img_filename_training = 'C://develop//python//UFL//data//train-images-idx3-ubyte';	
	mnist_lbl_filename_training = 'C://develop//python//UFL//data//train-labels-idx1-ubyte';
	
	debug 					= 1;
	imWidth					= 28;
	imHeight				= 28;
	imageChannels			= 1;
	numImages_unlabeled		= 30000;
	numImages_training		= 5000;
	numImages_test			= 10000;
	patchWidth				= 9;
	patchHeight				= 9;
	numPatches		 		= 60000;
	inputDim_patch			= patchWidth * patchHeight * imageChannels;
	inputDim_img			= imWidth * imHeight * imageChannels;
	numFeatures	 			= 32;
	nClasses				= 10;
	epsilon					= 1e-2;
	lambd					= 0.99;
	poolDim					= 5;
	
	#-------------------------
	#       Load Data
	#-------------------------
	if debug: print "Loading data..."
	
	# Read data from file
	numImages = numImages_unlabeled + numImages_training + numImages_test;
	
	images = DataInputOutput.loadMNISTImages(mnist_img_filename_training, numImages);
	images = np.reshape(images, [imHeight, imWidth, images.shape[1]]);
	images_unlabeled = images[:,:,0:numImages_unlabeled];
	images_training =  images[:,:,numImages_unlabeled:numImages_unlabeled+numImages_training];
	images_test = images[:,:,numImages_unlabeled+numImages_training:numImages_unlabeled+numImages_training+numImages_test];
	labels = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, numImages);
	labels_training =  labels[numImages_unlabeled:numImages_unlabeled+numImages_training];
	labels_test =  labels[numImages_unlabeled+numImages_training:numImages_unlabeled+numImages_training+numImages_test];
	
	# Sample patches
	patches = DataInputOutput.samplePatches(images_unlabeled, patchWidth, patchHeight, numPatches);
	
	# Normalize data: ZCA whiten patches
	patches = patches/255.0;
	instance_pca = PCA.PCA(inputDim_patch, 0.99, debug);
	ZCAwhite = instance_pca.computeZCAWhiteningMatrix(patches);
	patches_ZCAwhite = instance_pca.doZCAWhitening(patches);

	# Each patch should be normalized as x / ||x||_2 where x is the vector representation of the patch
	patches_ZCAwhite = DataNormalization.normL2(patches_ZCAwhite, axis=0)

	#-------------------------
	#     Learn Features
	#-------------------------
	if debug: print "Learning SoftICA features..."
	
	sizeLayers = [inputDim_patch, numFeatures];
	
	sica = SoftICA.SoftICA(sizeLayers, lambd, epsilon, debug=debug);
	
	success = sica.optimizeParameters(patches_ZCAwhite);
	
	weights = sica.getWeights();
	
	# Visualize the learned bases
	if debug>1:
		Visualization.displayNetwork(np.transpose(weights));
		
	#-------------------------
	#     Extract Features
	#-------------------------
	if debug: print "Extracting features..."
	
	# Pre-multiply the weights with whitening matrix, equivalent to whitening each image patch before applying convolution.
	weights = np.dot(weights, ZCAwhite);
	# Reshape SoftICA weights to be convolutional weights.
	weights = np.reshape(weights, [numFeatures, patchWidth, patchHeight]);
	weights = np.transpose(weights, [2,1,0]);

	activations_training = convolveAndPool(images_training, weights, poolDim);
	activations_test = convolveAndPool(images_test, weights, poolDim);
	
	if 0:
		for i in range(activations_training.shape[2]):
			pl.figure()
			pl.imshow(activations_training[:,:,i,0], cmap='gray');
			pl.show();
	
	featureDim = activations_training.shape[0] * activations_training.shape[1] * activations_training.shape[2];

	features_training = np.reshape(activations_training, [featureDim, activations_training.shape[3]])
	features_test = np.reshape(activations_test, [featureDim, activations_test.shape[3]])

	#-------------------------
	# Train Softmax Classifier
	#-------------------------
	if debug: print "Learning classification model..."
	
	softmaxModel = Softmax.Softmax(featureDim, nClasses, debug);
	
	success = softmaxModel.optimizeParameters(features_training, labels_training);
	
	#-------------------------
	# Testing
	#-------------------------
	if debug: print "Testing..."
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(softmaxModel.predict(features_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == np.argmax(softmaxModel.predict(features_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	