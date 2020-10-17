''' Softmax.py
	
	Implementation of Softmax regression model
	
	Author: Cem Karaoguz
	Date: 26.02.2015
	Version: 1.0
'''

import numpy as np
from scipy.optimize import minimize

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions

class Softmax:
	''' 
	Softmax regression class
	'''

	def __init__(self,
	             nParams,
				 nClasses,
				 debug=0):
		''' 
		Initialization function of the Softmax regression class
		
		Arguments
		nParams		: number of parameters (input dimensions)
		nClasses	: number of classes to identify
		debug		: debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.nParams = nParams;
		self.nClasses = nClasses;
		
		self.theta = np.random.rand(self.nClasses*self.nParams)*0.001;
		
		self.thetaMatrixPrototype = [self.nClasses, self.nParams]
		
		if debug:
			print 'DEBUG:Softmax:init: initialized for nParams: ', self.nParams;
			print 'DEBUG:Softmax:init: initialized for nClasses: ', self.nClasses;
			print
		
		self.isInitialized = True;
	
	def rollParameters(self, theta):
		''' 
		Converts a given parameter matrix into a vector
		
		Arguments
		theta	: parameter matrix
		
		Returns
		theta	: parameter vector
		'''
		assert self.isInitialized, 'ERROR:Softmax:rollParameters: The instance is not properly initialized'
		assert np.shape(theta)[0]==self.thetaMatrixPrototype[0], 'ERROR:Softmax:rollParameters: Dimensions of given parameters do not match the internal structure'
		assert np.shape(theta)[1]==self.thetaMatrixPrototype[1], 'ERROR:Softmax:rollParameters: Dimensions of given parameters do not match the internal structure'
		
		return theta.flatten();
		
	def unrollParameters(self, theta):
		''' 
		Converts the vectorized parameters into matrix
		
		Arguments
		theta	: parameter vector
		
		Returns
		theta	: parameter matrix
		'''
		assert self.isInitialized, 'ERROR:Softmax:unrollParameters: The instance is not properly initialized'
		assert len(theta)==self.thetaMatrixPrototype[0]*self.thetaMatrixPrototype[1], 'ERROR:Softmax:unrollParameters: dimensions of given parameters do not match internal parameter structure'
		
		return np.reshape(theta, self.thetaMatrixPrototype);
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the Softmax regression objective function for given parameters
		(theta), data matrix (X) and corresponding labels (y):
		
		f = -( Y * log( P(Y|X;theta) ) )
		
		where Y is ground truth matrix, a binary matrix where for each column (i.e. sample) 
		the row corresponding to the true class is one and the rest is zero
		
		P(Y|X;theta) = exp(theta'*X)/sum_j(exp(theta_j'*X)),	j = 1 to number of classes
		

		Arguments
		theta	: function parameters in the form (number of parameters * number of classes, )
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:Softmax:computeCost: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:computeCost: Dimensions of given data do not match with the number of parameters'
		
		epsilon = 1e-6;
		
		theta = self.unrollParameters(theta);
		
		f = 0;
		
		nSamples = X.shape[1];
		
		aux1 = np.exp(np.dot(theta, X));
		P = AuxFunctions.doUnbalancedMatrixOperation(aux1, np.sum(aux1, 0), 'div', axis=0);
		
		# Guard for log(0)
		if np.min(P)<epsilon:
			P = P + epsilon;
		aux3 = np.transpose(np.log(P));
		
		#aux3 = np.transpose(np.log(P.clip(min=epsilon)));
		
		
		aux4 = np.repeat(np.reshape(range(self.nClasses), [1, self.nClasses]), nSamples, 0)
		aux5 = np.repeat(np.reshape(y, [nSamples, 1]), self.nClasses, 1);
		
  		f = (-1.0/nSamples) * np.sum(aux3[aux4==aux5]);
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the Softmax regression objective function wrt parameters
		(theta) for a given data matrix (X) and corresponding labels (y):
		
		g = -( X * (Y - P(y|X;theta)) )
		
		where Y is ground truth matrix, a binary matrix where for each column (i.e. sample) 
		the row corresponding to the true class is one and the rest is zero
		
		P(Y|X;theta) = exp(theta'*X)/sum_j(exp(theta_j'*X)),	j = 1 to number of classes
		
		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (number of parameters*number of classes,)
		'''
		assert self.isInitialized, 'ERROR:Softmax:computeGradient: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:computeGradient: Dimensions of given data do not match with the number of parameters'
		
		theta = self.unrollParameters(theta);
		
		g = np.zeros(np.shape(theta));
		
		nSamples = X.shape[1];
		
		aux1 = np.exp(np.dot(theta, X));
		P = AuxFunctions.doUnbalancedMatrixOperation(aux1, np.sum(aux1, 0), 'div', axis=0);
		aux4 = np.repeat(np.reshape(range(self.nClasses), [1, self.nClasses]), nSamples, 0)
		aux5 = np.repeat(np.reshape(y, [nSamples, 1]), self.nClasses, 1);
		aux6 = aux4==aux5;
		
		g = (-1.0/nSamples) * np.transpose(np.dot(X, np.transpose(np.transpose(aux6.astype(int)) - P)));
		
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
		assert self.isInitialized, 'ERROR:Softmax:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:Softmax:testGradient: Testing gradient computation...'
		
		result = 0;
		
		grad = self.computeGradient(self.theta, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.theta, args=(X, y) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:Softmax:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Softmax:testGradient: Gradient check PASSED!'
				print
			
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:Softmax:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Softmax:testGradient: Gradient check FAILED!'
				print
				
			result = -1;
			
		return result;
	
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the Softmax regression model
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:Softmax:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:optimizeParameters: Dimensions of given data do not match with the number of parameters'

		if self.debug: print "DEBUG:Softmax:optimizeParameters: Optimizing parameters..."
		
		# Set optimization options
		method = 'L-BFGS-B'
		options = {};
		options['maxiter'] = 100;
		
		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.theta, args=(X, y), method=method, options=options)
		
		# Set the new values
		self.theta = result.x;
		
		if self.debug: print "DEBUG:Softmax:optimizeParameters: Optimization result: ", result.message
		
		return result.success;

	def doForwardPropagation(self, X, weights):
		''' 
		Computes the forward propagation of the input in the network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		weights		: weight matrix to be used with forward propagation
		
		Returns
		output		: output of the Softmax model given via weights
		'''
		assert self.isInitialized, 'ERROR:Softmax:doForwardPropagation: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:doForwardPropagation: Dimensions of given data do not match with the number of parameters'
		assert np.shape(weights)[0]==self.thetaMatrixPrototype[0], 'ERROR:Softmax:doForwardPropagation: Dimensions of given weights do not match the internal structure'
		assert np.shape(weights)[1]==self.thetaMatrixPrototype[1], 'ERROR:Softmax:doForwardPropagation: Dimensions of given weights do not match the internal structure'
		
		return np.dot(weights, X)

	def predict(self, X):
		'''
		Applies the Softmax regression model to the given data
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		
		Returns
		pred	: prediction, matrix of floating points in the form [number of classes, number of samples]
		'''
		assert self.isInitialized, 'ERROR:Softmax:predict: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:predict: Dimensions of given data do not match with the number of parameters'
		
		theta = self.unrollParameters(self.theta);
		pred = np.dot(theta, X);
		
		return pred
		
	def getWeights(self):
		''' 
		Returns the Softmax model parameters in matrix form
		'''
		assert self.isInitialized, 'ERROR:Softmax:getWeights: The instance is not properly initialized'
		
		return self.unrollParameters(self.theta);

	def setWeights(self, theta):
		''' 
		Updates the Softmax model parameters with a given parameter matrix
		'''
		assert self.isInitialized, 'ERROR:Softmax:setWeights: The instance is not properly initialized'
		assert len(np.shape(theta))==2, 'ERROR:Softmax:setWeights: Dimensions of given parameters do not match with internal structure'
		assert np.shape(theta)[0]==self.nClasses, 'ERROR:Softmax:setWeights: Dimensions of given parameters do not match with internal structure'
		assert np.shape(theta)[1]==self.nParams, 'ERROR:Softmax:setWeights: Dimensions of given parameters do not match with internal structure'
		
		self.theta = self.rollParameters(theta);

	def getParameters(self):
		''' 
		Returns the Softmax model parameters in unstacked form
		'''
		assert self.isInitialized, 'ERROR:Softmax:getParameters: The instance is not properly initialized'
		
		return self.theta;
	
	def getParameterSize(self):
		'''
		Returns the size of model parameters
		'''
		assert self.isInitialized, 'ERROR:Softmax:getParameterSize: The instance is not properly initialized'
		
		return self.thetaMatrixPrototype[0] * self.thetaMatrixPrototype[1]

	def getParameters_asLayer(self):
		''' 
		Wrapper function for getParameters for cases where Softmax model is
		used as a layer of a deep network.
		'''
		assert self.isInitialized, 'ERROR:Softmax:getParameters_asLayer: The instance is not properly initialized'
		
		return self.getParameters();

	def getParameterSize_asLayer(self):
		'''
		Wrapper function for getParameterSize for cases where Softmax model is
		used as a layer of a deep network.
		'''
		assert self.isInitialized, 'ERROR:Softmax:getParameterSize_asLayer: The instance is not properly initialized'

		return self.getParameterSize()

	def setParameters_asLayer(self, theta):
		''' 
		Wrapper function for setWeights for cases where Softmax model is
		used as a layer of a deep network.
		'''
		assert self.isInitialized, 'ERROR:Softmax:setParameters_asLayer: The instance is not properly initialized'
		assert len(theta)==self.thetaMatrixPrototype[0]*self.thetaMatrixPrototype[1], 'ERROR:Softmax:setParameters_asLayer: dimensions of given parameters do not match internal parameter structure'

		self.theta = theta;
		
	def doForwardPropagation_asLayer(self, X, theta):
		''' 
		Wrapper function for doForwardPropagation for cases where Softmax model is
		used as a layer of a deep network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		theta		: model parameters for the first layer, must be packed as [weights+biases]
		
		Returns
		activation	: activation if the first layer
		'''
		assert self.isInitialized, 'ERROR:Softmax:doForwardPropagationAsLayer: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Softmax:doForwardPropagation: Dimensions of given data do not match with the number of parameters'
		assert np.size(theta)==self.thetaMatrixPrototype[0]*self.thetaMatrixPrototype[1], 'ERROR:Softmax:doForwardPropagation: Dimensions of given weights do not match the internal structure'
		
		weights = self.unrollParameters(theta);
		
		activation = self.doForwardPropagation(X, weights);
		
		# Convert output to probabilities:
		aux2 = AuxFunctions.doUnbalancedMatrixOperation(activation, np.amax(activation, 0), 'sub', axis=0); #Substracts the maximm value of the matrix "aux".
		aux3 = np.exp(aux2);
		y = AuxFunctions.doUnbalancedMatrixOperation(aux3, np.sum(aux3, 0), 'div', axis=0); #I divides the vector "aux3" by the sum of its elements.

		return y;
		
	def doBackPropagateError_asLayer(self, error, theta, layer_in, layer_out):
		'''
		'''
		weights = np.reshape(theta, self.thetaMatrixPrototype);
		delta = error;
		
		grad = np.transpose(np.dot(layer_in, np.transpose(delta)));
		
		error_prop = ( (np.dot(np.transpose(weights), error)));
		
		return grad.flatten(), error_prop;

	
if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Digit classification (0 to 9) using Softmax regression and images from the MNIST data set
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
	nSamples_training = 20000;
	nSamples_test = 30000;
	nClasses = 10;
	
	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_training);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_training);
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_test);
	
	dataDim, nSamples_training = np.shape(images_training);
	dataDim, nSamples_test = np.shape(images_test);
	
	# Normalize data 
	images_training = images_training / 255.0;
	images_test = images_test / 255.0;
	images_training = DataNormalization.normMeanStd( images_training );
	images_test = DataNormalization.normMeanStd( images_test );
	
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
		
	softmaxregressor = Softmax(inputDim, nClasses, debug);
	
	if debug:
		# Check if the gradient computation is OK on a smaller subset of data
		softmaxregressor.testGradient(images_training[:,0:20], labels_training[0:20])
		
	success = softmaxregressor.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(softmaxregressor.predict(images_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == np.argmax(softmaxregressor.predict(images_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	