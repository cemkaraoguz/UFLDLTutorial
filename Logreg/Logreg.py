''' Logreg.py
	
	Implementation of logistic regression model
	
	Author: Cem Karaoguz
	Date: 20.02.2015
	Version: 1.0
'''

import numpy as np
from scipy.optimize import minimize
import pylab as pl

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions

class Logreg:
	''' 
	Logistic regression class
	'''

	def __init__(self, nParams, debug=0):
		''' 
		Initialization function of the logistic regression class
		
		Arguments
		nParams	: number of parameters (input dimensions)
		debug	: debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.nParams = nParams;
		
		self.theta = np.random.rand(self.nParams, 1)*0.001;
		
		if debug:
			print 'DEBUG:Logreg:init: initialized for nParams: ', self.nParams;
			print
		
		self.isInitialized = True;
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the logistic regression objective function for given parameters
		(theta), data matrix (X) and corresponding labels (y) following:
		
		f = -( y * log(h(X|theta)) + (1-y)*log(1-h(X|theta)) )
		
		where
		
		h(X|theta) = 1/(1 + exp(-theta'X))

		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		f		: computed cost (floating point number)
		'''
		f = 0;
		
		h = 1.0/(1.0 + np.exp(-1.0 * np.dot(theta.T, X)));
		f = -1 * np.sum( (y * np.log(h)) + ((1-y) * np.log(1-h)) );

		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the logistic regression objective function wrt parameters
		(theta) for a given data matrix (X) and corresponding labels (y) following:
		
		g = -( X * (h(X|theta) - y) )
		
		where
		
		h(X|theta) = 1/(1 + exp(-theta'X))

		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (number of parameters,)
		'''
		g = np.zeros(np.shape(theta));
		
		h = 1.0/(1.0 + np.exp(-1.0 * np.dot(theta.T, X)));
		g = np.dot(X, np.transpose(h - y));
		
		return g.flatten()
	
	def testGradient(self, X, y):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:Logreg:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Logreg:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:Logreg:testGradient: Testing gradient computation...'
		
		result = 0;
		
		grad = self.computeGradient(self.theta, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.theta, args=(X, y) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:Logreg:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Logreg:testGradient: Gradient check PASSED!'
				print
			
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:Logreg:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Logreg:testGradient: Gradient check FAILED!'
				print
			
			result = -1;
				
		return result
			
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the logistic regression model
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:Logreg:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Logreg:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print "DEBUG:Logreg:optimizeParameters: Optimizing parameters..."
		
		# Set optimization options
		method = 'L-BFGS-B'
		options = {};
		options['maxiter'] = 100;
		
		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.theta, args=(X, y), method=method, options=options)
		
		# Set the new values
		self.theta = np.reshape(result.x, [self.nParams, 1]);
		
		if self.debug: print "DEBUG:Logreg:optimizeParameters: Optimization result: ", result.message
		
		return result.success;

	def predict(self, X):
		'''
		Applies the logistic regression model to the given data
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		
		Returns
		pred	: prediction vector in the form of [1, number of samples]
		'''
		assert self.isInitialized, 'ERROR:Logreg:predict: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Logreg:predict Dimensions of given data do not match with the number of parameters'
		
		pred = np.dot(self.theta.T, X);
		
		return pred
		
if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Binary digit classification (i.e. zeros and ones) using logistic regression and images from the MNIST data set
	# --------------------------
	if 1:
	  mnist_lbl_filename_training = '/home/cem/develop/UFL/data/train-labels-idx1-ubyte';
	  mnist_img_filename_training = '/home/cem/develop/UFL/data/train-images-idx3-ubyte';
	  mnist_lbl_filename_test = '/home/cem/develop/UFL/data/t10k-labels-idx1-ubyte';
	  mnist_img_filename_test = '/home/cem/develop/UFL/data/t10k-images-idx3-ubyte';
	else:
	  mnist_lbl_filename_training = 'C://develop//python//UFL//data//train-labels-idx1-ubyte';
	  mnist_img_filename_training = 'C://develop//python//UFL//data//train-images-idx3-ubyte';
	  mnist_lbl_filename_test = 'C://develop//python//UFL//data//t10k-labels-idx1-ubyte';
	  mnist_img_filename_test = 'C://develop//python//UFL//data//t10k-images-idx3-ubyte';
	
	debug = 1;
	nSamples_training = 50000;
	nSamples_test = 50000;
	
	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_training);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_training);
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_test);
	
	# Take only the binary digits
	images_training = np.hstack( (images_training[:,labels_training==0], images_training[:,labels_training==1]) );
	labels_training = np.hstack( (labels_training[labels_training==0], labels_training[labels_training==1]) );
	images_test = np.hstack( (images_test[:,labels_test==0], images_test[:,labels_test==1]) );
	labels_test = np.hstack( (labels_test[labels_test==0], labels_test[labels_test==1]) );
	
	dataDim, nSamples_training = np.shape(images_training);
	dataDim, nSamples_test = np.shape(images_test);
	
	# Normalize data 
	images_training = images_training / 255.0;
	images_test = images_test / 255.0;
	images_training = DataNormalization.normMeanStd( images_training );
	images_test = DataNormalization.normMeanStd( images_test );
	
	# Shuffle examples.
	randind = np.random.permutation(nSamples_training);
	images_training = images_training[:, randind];
	labels_training = labels_training[randind];
	
	randind = np.random.permutation(images_test.shape[1]);
	images_test = images_test[:, randind];
	labels_test = labels_test[randind];
	
	if 0:
		pl.figure();
		sampleImage = np.reshape(images_training[:,0], [28, 28]);
		pl.imshow(sampleImage, cmap='gray');
		pl.show();
	
	# Include a row of 1s as an additional intercept feature.
	images_training = np.vstack( (np.ones((1, images_training.shape[1])), images_training) );
	images_test = np.vstack( (np.ones((1, images_test.shape[1])), images_test) );

	inputDim = images_training.shape[0];
	
	if debug:
		print 'Number of training samples: ', nSamples_training
		print 'Number of test samples: ', nSamples_test
		print 'Data dimensions: ', dataDim
		print 'Input dimensions: ', inputDim

	logisticregressor = Logreg(inputDim, debug);
	
	if debug:
		# Check if the gradient computation is OK on a smaller subset of data
		logisticregressor.testGradient(images_training[:,0:20], labels_training[0:20])
		
	success = logisticregressor.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == (logisticregressor.predict(images_training) > 0.5)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == (logisticregressor.predict(images_test) > 0.5)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'