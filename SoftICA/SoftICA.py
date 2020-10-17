''' SoftICA.py
	
	Implementation of Independent Component Analysis with reconstruction constraint
	
	Author: Cem Karaoguz
	Date: 13.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization
from UFL.PCA import PCA

class SoftICA:
	''' 
	Independent Component Analysis with soft reconstruction constraint
	
	Implementation is similar to Sparse Autoencoder network where the input 
	data is projected on a lower dimensional space of bases which are then 
	projected back to the output layer with the dimension same as the original data.
	
	Output of the network is the same as the input (i.e. reconstructive)
	Cost function is the L1 norm between the input and output layers with an
	additional constraint of the orthogonality of basis vectors.
	'''

	def __init__(self,
	             sizeLayers,
				 lambd=0.99,
				 epsilon=1e-2,
				 debug=0):
		''' 
		Initialization function of the SoftICA class
		
		Arguments
		sizeLayers		: Size of the layers, must be in the form [Input dimensions, hidden layer dimensions, output layer dimensions]
						  where output layer dimensions = input layer dimensions
		lambd			: Sparsity cost, default is 0.99
		epsilon			: L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon), default is 1e-2
		debug			: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = sizeLayers[0];
		self.featureDim = sizeLayers[1];
		self.lambd = lambd;
		self.epsilon = epsilon;
		
		assert self.inputDim>0, 'ERROR:SoftICA:init: Input size must be >0'
		assert self.featureDim>0, 'ERROR:SoftICA:init: Feature size must be >0'
		
		weights = np.random.rand(self.featureDim, self.inputDim)*0.01;
		weights = AuxFunctions.doUnbalancedMatrixOperation(weights, np.sqrt(np.sum(weights**2, 1)), 'div');
		
		self.params = weights.flatten();
		
		self.weightPrototype = (self.featureDim, self.inputDim);
		
		if debug:
			print 'DEBUG:SoftICA:init: initialized for inputDim: ', self.inputDim;
			print 'DEBUG:SoftICA:init: initialized for featureDim: ', self.featureDim;
			print 'DEBUG:SoftICA:init: initialized for lambda: ', self.lambd;
			print 'DEBUG:SoftICA:init: initialized for epsilon: ', self.epsilon;
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
		assert self.isInitialized, 'ERROR:SoftICA:unrollParameters: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters([theta], [self.weightPrototype]), 'ERROR:SoftICA:rollParameters: Weight dimension does not match the network topology' ;
		
		return theta.flatten();
		
	def unrollParameters(self, theta):
		''' 
		Converts the vectorized parameters into matrix
		
		Arguments
		theta	: parameter vector
		
		Returns
		theta	: parameter matrix
		'''
		assert self.isInitialized, 'ERROR:SoftICA:unrollParameters: The instance is not properly initialized'
		assert len(theta)==self.featureDim*self.inputDim, 'ERROR:SoftICA:unrollParameters: dimensions of given parameters do not match internal parameter structure'
		
		return np.reshape(theta, [self.featureDim, self.inputDim]);
		
	def getWeights(self):
		''' 
		Returns the SoftICA weight matrix
		'''
		return self.unrollParameters(self.params)
		
	def computeCost(self, theta, X):
		''' 
		Computes the value of the ICA objective function with reconstruction term
		for given features (theta) and data matrix (X):
		
		f = lambda * || theta*X ||_1 + 1/2(|| theta'*theta*X - X ||_2)^2
		
		where || . ||_k denotes Lk norm: : sum_i(|(x_i)^k|)^(1/k)
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:SoftICA:computeCost: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:SoftICA:computeCost: Dimensions of given data do not match with the number of parameters'
		
		f = 0;
		
		nSamples = X.shape[1];
		
		# Unpack weight matrix
		weightMatrix = self.unrollParameters(theta);
		
		# Project weights to norm ball (prevents degenerate bases)
		weightMatrix_old = weightMatrix;
		weightMatrix = self.l2rowscaled(weightMatrix, 1);

		aux1 = np.sqrt((np.dot(weightMatrix, X)**2) + self.epsilon);
		aux2 = np.dot(np.dot(np.transpose(weightMatrix), weightMatrix), X) - X;
		
		f = 1.0/nSamples * (self.lambd * np.sum(aux1) + np.sum(aux2**2));
		
		return f
		
	def computeGradient(self, theta, X):
		''' 
		Computes gradients of the ICA objective function with reconstruction term 
		for given features (theta) and data matrix (X):
		
		g = (theta*X)/sqrt(theta*X + epsilon)*X' + theta*(theta'*theta*X - X)*X' + theta*X*(theta'*theta*X - X)'
		
		Arguments
		theta	: function parameters in the form [feature dim, input dim]
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (feature dim * input dim,)
		'''
		assert self.isInitialized, 'ERROR:SoftICA:computeGradient: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:SoftICA:computeCost: Dimensions of given data do not match with the number of parameters'
		
		nSamples = X.shape[1];

		# Unpack weight matrix		
		weightMatrix = self.unrollParameters(theta);
	
		# Project weights to norm ball (prevents degenerate bases)
		weightMatrix_old = weightMatrix;
		weightMatrix = self.l2rowscaled(weightMatrix, 1);

		aux1 = np.sqrt((np.dot(weightMatrix, X)**2) + self.epsilon);
		aux2 = np.dot(np.dot(np.transpose(weightMatrix), weightMatrix), X) - X;
		
		
		grad_term1 = self.lambd * 1.0/nSamples * np.dot((np.dot(weightMatrix, X)/aux1), np.transpose(X));
		grad_term2 = 2.0/nSamples * (np.dot(np.dot(weightMatrix, aux2), np.transpose(X)) + np.dot(np.dot(weightMatrix, X), np.transpose(aux2)));
		
		Wgrad = grad_term1 + grad_term2;

		# Back project gradient for minFunc
		grad = self.l2rowscaledg(weightMatrix_old, weightMatrix, Wgrad, 1);
		
		return grad.flatten();
	
	def l2rowscaled(self, x, alpha):
		'''
		Project weights to norm ball to prevent degenerate bases
		
		Arguments
		x		: Weight matrix
		alpha	: Scale factor
		
		Returns
		y		: Weight matrix projected to norm ball
		'''
		normeps = 1e-5;
		epssumsq = np.sum(x**2, 1) + normeps;   

		l2rows = np.sqrt(epssumsq) * alpha;
		y = AuxFunctions.doUnbalancedMatrixOperation(x, l2rows, 'div');
		
		return y
		
	def l2rowscaledg(self, x, y, outderv, alpha):
		'''
		Back-projects weight gradients from norm ball to their original space
		
		Arguments
		x		: Old weight matrix
		y		: Weight matrix projected to norm ball
		outderv	: Gradients projected to norm ball
		alpha	: Scale factor
		
		Returns
		Weight gradient matrix back-projected to the original weight space
		'''
		normeps = 1e-5;
		epssumsq = np.sum(x**2, 1) + normeps;	

		l2rows = np.sqrt(epssumsq) * alpha;

		if len(y)==0:
			y = AuxFunctions.doUnbalancedMatrixOperation(x, l2rows, 'div');
			
		aux1 = AuxFunctions.doUnbalancedMatrixOperation(outderv, l2rows, 'div');
		aux2 = AuxFunctions.doUnbalancedMatrixOperation(y, (np.sum(outderv * x, 1) / epssumsq), 'mul');
		
		return aux1 - aux2
	   
	def testGradient(self, X):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:SoftICA:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:SoftICA:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:SoftICA:testGradient: Testing gradient computation...'
		
		result = 0;
		
		grad = self.computeGradient(self.params, X);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.params, args=((X,)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:SoftICA:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SoftICA:testGradient:Gradient check PASSED!'
				print
				
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:SoftICA:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SoftICA:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
	
		return result
	
	def optimizeParameters(self, X):
		'''
		Optimize for the orthonormal ICA objective with soft orthogonality
		constraint enforced via reconstruction term.
		
		Arguments
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:SoftICA:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:SoftICA:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:SoftICA:optimizeParameters: Optimizing parameters...'
		
		# Set optimization options
		method = 'L-BFGS-B'
		options = {};
		options['maxiter'] = 300;

		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.params, args=(X,), method=method, options=options)
		
		# Set the new values
		self.params = result.x;
		
		if self.debug: print 'DEBUG:SoftICA:optimizeParameters: Optimization result: ', result.message
		
		return result.success

if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Learning orthagonal bases of images of handwritten digits (MNIST dataset)
	# --------------------------

	if 1:
	  mnist_img_filename_training = '/home/cem/develop/UFL/data/train-images-idx3-ubyte';
	else:
	  mnist_img_filename_training = 'C://develop//python//UFL//data//train-images-idx3-ubyte';	
	  
	doTest 			= True;			# Test gradient computation?
	debug 			= 1;
	numImages		= 10000
	imWidth			= 28;
	imHeight		= 28;
	numPatches 		= 10000;
	imageChannels	= 1;
	patchWidth		= 9;
	patchHeight		= 9;
	inputDim		= patchWidth * patchHeight * imageChannels;
	numFeatures	 	= 50;
	epsilon			= 1e-2;
	lambd			= 0.99;

	# Read data from file
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, numImages);
	images_training = np.reshape(images_training, [imHeight, imWidth, images_training.shape[1]]);
	
	# Sample patches
	patches = DataInputOutput.samplePatches(images_training, patchWidth, patchHeight, numPatches);
	
	if debug>1:
		Visualization.displayNetwork(patches[:, 0:100]);
	
	# Normalize data: ZCA whiten patches
	patches = patches/255.0;
	instance_pca = PCA.PCA(inputDim, 0.99, debug);
	patches_ZCAwhite = instance_pca.doZCAWhitening(patches);

	# Each patch should be normalized as x / ||x||_2 where x is the vector representation of the patch
	patches_ZCAwhite = DataNormalization.normL2(patches_ZCAwhite, axis=0)

	if debug:
		print 'Number of samples: ', patches.shape[1];
		
	if doTest:

		inputDim_test = 3;
		numFeatures_test = 6;
		numPatches_test	 = 10;
		
		sizeLayers = [inputDim_test, numFeatures_test];
	
		patches_test = np.random.rand(inputDim_test, numPatches_test);
		
		sica_test = SoftICA(sizeLayers, debug=2);
	
		print 'Checking gradient...';
		
		sica_test.testGradient(patches_test);

	sizeLayers = [inputDim, numFeatures];
	
	sica = SoftICA(sizeLayers, lambd, epsilon, debug=debug);
	
	success = sica.optimizeParameters(patches_ZCAwhite);
	
	# Visualize the learned bases
	weightMatrix = sica.getWeights();
	Visualization.displayNetwork(np.transpose(weightMatrix));