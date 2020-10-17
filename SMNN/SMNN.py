''' SMNN.py
	
	Implementation of Supervised Multilayer Neural Network
	
	Author: Cem Karaoguz
	Date: 27.02.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions

SMNN_ACTFUN_SIGMOID = 0;
SMNN_ACTIVATION_FUNCTIONS = [SMNN_ACTFUN_SIGMOID];

class SMNN:
	''' 
	Supervised Multilayer Neural Network
	'''

	def __init__(self,
	             dimLayers,
				 lambd=0,
				 activation_fun=SMNN_ACTFUN_SIGMOID,
				 debug=0):
		''' 
		Initialization function of the Supervised Multilayer Neural Network class
		
		Arguments
		dimLayers		: size of the layers, must be in the form [Input layer dim., hidden layer 1 dim., hidden layer 2 dim., ..., output layer dim.]
		lambd			: scaling parameter for l2 weight regularization penalty, default is 0
		activation_fun	: Activation function for neurons, possible values [SMNN_ACTFUN_SIGMOID*]
		debug			: debugging flag
		'''
		self.isInitialized = False;
		
		assert activation_fun in SMNN_ACTIVATION_FUNCTIONS, 'ERROR:SMNN:init: activation function not recognized'
		assert lambd>=0, 'ERROR:SMNN:init: lambda should be >=0'
		
		self.debug = debug;
		self.dimLayers = dimLayers;
		self.lambd = lambd;
		self.activation_fun = activation_fun;
		
		self.nLayers = len(dimLayers);
		
		assert self.nLayers>2, 'ERROR:SMNN:init: Layer size must be minimum three: input-hidden-output'
		
		weights = [];
		biases = [];
		self.weightPrototypes = [];
		self.biasPrototypes = [];
		self.sizeParams = 0;
		for i in range(self.nLayers - 1):
			# Xavier's scaling factor from X. Glorot, Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. AISTATS 2010.
			s = np.sqrt(6.0) / np.sqrt(self.dimLayers[i+1] + self.dimLayers[i]);
			# Set random weights
			weights.append( np.random.rand(self.dimLayers[i+1], self.dimLayers[i])*2*s - s );
			biases.append( np.zeros((self.dimLayers[i+1], 1)) );
			# Set network topology
			self.weightPrototypes.append((self.dimLayers[i+1], self.dimLayers[i]));
			self.biasPrototypes.append((self.dimLayers[i+1], 1));
			# Total number of parameters
			self.sizeParams = self.sizeParams + (self.dimLayers[i+1] * self.dimLayers[i]) + (self.dimLayers[i+1]);
		
		self.params = self.rollParameters(weights, biases);
		
		if debug:
			print 'DEBUG:SMNN:init: initialized for nLayers: ', self.nLayers;
			print 'DEBUG:SMNN:init: initialized for dimLayers: ', self.dimLayers;
			print
		
		self.isInitialized = True;
		
	def rollParameters(self, weights, biases):
		''' 
		Converts the parameters in matrix form into vector
		
		Arguments
		weights	: list of weight matrices of each layer 
		biases	: list of bias vectors of each layer 
		
		Returns
		params	: parameter vector
		'''
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:SMNN:rollParameters: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:SMNN:rollParameters: bias dimension does not match the network topology';
		
		params = np.array([]);
		for i in range(len(weights)):
			params = np.hstack((params, weights[i].flatten(), biases[i].flatten()))
		
		return params
		
	def unrollParameters(self, params):
		''' 
		Converts the vectorized parameters into list of matrices
		
		Arguments
		params	: parameter vector
		
		Returns
		weights	: list of weight matrices of each layer 
		biases	: list of bias vectors of each layer 
		'''
		assert len(params)==self.sizeParams, 'ERROR:SMNN:unrollParameters: Parameter size mismatch'
		
		weights = [];
		biases = [];
		read_start = 0;
		read_end = 0;
		
		for i in range(self.nLayers - 1):
			# set the end index for read
			read_end = read_start + self.dimLayers[i+1]*self.dimLayers[i];
			# read the weights for the current layer
			w = params[read_start:read_end];
			# reshape and the weights
			weights.append( np.reshape(w, (self.dimLayers[i+1], self.dimLayers[i])) );
			# set the start index for the next read
			read_start = read_end;
			# set the end index for the next read
			read_end = read_start + self.dimLayers[i+1];
			# read the bias terms
			b = params[read_start:read_end];
			# reshape and store the bias
			biases.append( np.reshape(b, (self.dimLayers[i+1], 1)) )
			# set the start index for the next read
			read_start = read_end;
		
		return weights, biases;
	
	def doForwardPropagation(self, X, weights, biases):
		''' 
		Computes the forward propagation of the input in the SMNN:
		
		Z{l+1} = W{l}*H{l} + B{l}
		H{l+1} = f(Z{l+1})
		
		where {l} and {l+1} denote layers,
		B is the bias matrix, columnwise repetition of the bias vector with the number of samples,
		Z is the output matrix of neurons before the activation function is applied,
		f(.) is the activation function
		H is the output matrix of neurons after the activation function is applied (h{1}=X),
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		weights		: list of weight matrices of each layer
		biases		: list of bias vectors of each layer
		
		Returns
		outputs		: list of output matrices (z) of each layer (output of neuron before activation function)
		activities	: list of activation matrices (h) of each layer (output of neuron after activation function)
		'''
		assert self.isInitialized, 'ERROR:SMNN:doForwardPropagation: The instance is not properly initialized'
		
		# Default behaviour is bad implementation
		#if len(weights)==0 or len(biases)==0:
		#	[weights, biases] = self.unrollParameters(self.params);
		
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:SMNN:doForwardPropagation: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:SMNN:doForwardPropagation: bias dimension does not match the network topology';
		
		outputs = [];
		activities = [];
		for layer in range(self.nLayers-1):
			
			if layer==0:
				x = X;
			else:
				x = activities[layer-1];
				
			z = np.dot(weights[layer], x) + np.repeat(biases[layer], x.shape[1], 1);
			
			if self.activation_fun==SMNN_ACTIVATION_FUNCTIONS[SMNN_ACTFUN_SIGMOID]:
				h = AuxFunctions.sigmoid(z);
			else:
				# Should not be here
				print 'ERROR:SMNN:doForwardPropagation: Wrong activation function'
				sys.exit()
				
			outputs.append(z);
			activities.append(h);
		
		return [outputs, activities];
		
	def doBackPropagation(self, error, activities, weights):
		''' 
		Computes the back propagation of the error in the SMNN:
		
		E_{l-1} = W_{l-1} * E_{l} * df(Z{l-1})/dz
		
		where {l} and {l-1} denote layers,
		E is the (propagated) error matrix,
		df(Z)/dz is the derivatives of the activation function at points Z.
		
		Arguments
		error		: error matrix of the output layer with columns corresponding to the samples, rows corresponding to the units
		activities	: list of activation matrices of each layer (output of neuron after activation function)
		weights		: list of weight matrices of each layer
		
		Returns
		deltas		: list of error matrices of each layer
		'''
		# Default behaviour is bad implementation		
		#if len(weights)==0:
		#	[weights, biases] = self.unrollParameters(self.params);
		
		assert self.isInitialized, 'ERROR:SMNN:doBackPropagation: The instance is not properly initialized'
		assert len(weights)==(self.nLayers-1), 'ERROR:SMNN:doBackPropagation: given weight matrix list do not match with the internal layer size'
		
		deltas = [];
		
		if self.activation_fun==SMNN_ACTIVATION_FUNCTIONS[SMNN_ACTFUN_SIGMOID]:
			
			# compute the delta for output layer
			d = -1.0 * error * activities[-1] * (1 - activities[-1]);
		
			deltas.append(d)
			
			# propagate the delta
			for l in range(self.nLayers-2):
				d = np.dot(np.transpose(weights[-1-l]), deltas[l] ) * activities[-2-l] * (1 - activities[-2-l]);
				deltas.append(d);
		else:
			# Should not be here
			print 'ERROR:SMNN:doBackPropagation: Wrong activation function'
			sys.exit()
		
		return list(reversed(deltas))
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the SMNN objective function for given features (theta),
		data matrix (X) and corresponding labels (y):
		
		f = 1/2 * sum((Y - H)^2) +  1/2 * lambda * sum(W^2)
		
		where
		
		Y is ground truth matrix, a binary matrix where for each column (i.e. sample) 
		the row corresponding to the true class is one and the rest is zero,
		H is the activity matrix with columns corresponding to the samples and rows 
		corresponding to the output layer units.
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:SMNN:computeCost: The instance is not properly initialized'
		
		f = 0;
		
		[weights, biases] = self.unrollParameters(theta);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);
		
		nSamples = X.shape[1];
		
		aux1 = np.repeat(np.reshape(range(self.dimLayers[-1]), [self.dimLayers[-1], 1]), nSamples, 1)
		aux2 = np.repeat(np.reshape(y, [1, nSamples]), self.dimLayers[-1], 0);
		aux3 = (aux1==aux2);
		
		W_sum = 0;
		for l in range(len(weights)):
			W_sum = W_sum + np.sum((weights[l])**2);
		
		f = np.sum( 0.5 * np.sum((aux3.astype(int) - activities[-1])**2, 1)) + ((self.lambd/2.0) * W_sum);
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the SMNN objective function for given parameters,	data and corresponding labels
		using the back propagation:
		
		dJ(W,b;X,y)/dW_{l} = E_{l+1} * H_{l}'
		dJ(W,b;X,y)/db_{l} = sum(E_{l+1})
		
		where sum(.) is taken columnwise i.e. over samples
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		grad	: gradients of weights and biases in rolled form
		'''
		nSamples = np.size(y);
		
		[weights, biases] = self.unrollParameters(theta);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);
		
		aux1 = np.repeat(np.reshape(range(self.dimLayers[-1]), [self.dimLayers[-1], 1]), nSamples, 1)
		aux2 = np.repeat(np.reshape(y, [1, nSamples]), self.dimLayers[-1], 0);
		aux3 = (aux1==aux2);
		errorOut = aux3.astype(int) - activities[-1];
		
		deltas = self.doBackPropagation(errorOut, activities, weights);
		
		gradients_W = [];
		gradients_b = [];
		for layer in range(self.nLayers-1):
			if layer==0:
				x_in = X;
			else:
				x_in = activities[layer-1];
			
			gradients_W.append( np.dot(deltas[layer], np.transpose(x_in)) );
			gradients_b.append( np.sum(deltas[layer], 1) );
		
		return self.rollParameters(gradients_W, gradients_b);
	
	def testGradient(self, X, y):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [input dim., number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:SMNN:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SMNN:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:SMNN:testGradient: Testing gradient computation...'
		
		result = 0;
		
		grad = self.computeGradient(self.params, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.params, args=(X, y) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:SMNN:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SMNN:testGradient:Gradient check PASSED!'
				print
				
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:SMNN:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SMNN:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
	
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the SMNN model
		
		Arguments
		X		: data in the form [input dim., number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:SMNN:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SMNN:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		assert X.shape[1]==len(y), 'ERROR:SMNN:optimizeParameters: Dimensions of given data and labels do not match';
		
		if self.debug: print "DEBUG:SMNN:optimizeParameters: Optimizing parameters..."
		
		# Set optimization options
		method = 'L-BFGS-B'
		options = {};
		options['maxiter'] = 300;

		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.params, args=(X, y), method=method, options=options)
		
		# Set the new values
		self.params = result.x;
		
		if self.debug: print "DEBUG:SMNN:optimizeParameters: Optimization result: ", result.message
		
		return result.success;

	def predict(self, X):
		'''
		Applies the SMNN model to the given data
		
		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		pred	: prediction matrix in the form [output dim., number of samples]
		'''
		assert self.isInitialized, 'ERROR:SMNN:predict: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SMNN:predict: Dimensions of given data do not match with the number of parameters'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);

		return activities[-1];
		
if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Learning a SMNN model for classifying images of handwritten digits (MNIST dataset)
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
	  
	doTest 				= True;					# Test gradient computation?
	debug 				= 1;
	imWidth				= 28;
	imHeight			= 28;
	inputDim			= imWidth * imHeight;
	nSamples_max_train 	= 20000;
	nSamples_max_test 	= 30000;
	nClasses			= 10;
	hiddenDim 			= 256;
	lambd 				= 0;					# scaling parameter for l2 weight regularization penalty
	activation_fun 		= SMNN_ACTFUN_SIGMOID	# Currently implemented: sigmoid 
	
	if doTest:
		nSamples_grtest = 10;
		inputDim_grtest = 4;
		hiddenDim_grtest = 5;
		outputDim_grtest = 2;
		
		dimLayers_grtest = [inputDim_grtest, hiddenDim_grtest, outputDim_grtest];
		
		data_grtest = np.random.rand(inputDim_grtest, nSamples_grtest);
		labels_grtest = np.random.randint(outputDim_grtest, size=nSamples_grtest);
		
		smnn_grtest = SMNN(dimLayers_grtest, debug=debug);
		
		smnn_grtest.testGradient(data_grtest, labels_grtest);
		

	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_max_train);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_max_train);
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_max_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_max_test);
	
	nSamples_training = np.shape(images_training)[1];
	nSamples_test = np.shape(images_test)[1];
	
	# Normalize data 
	images_training = images_training / 255.0;
	images_test = images_test / 255.0;
	images_training = DataNormalization.normMeanStd( images_training );
	images_test = DataNormalization.normMeanStd( images_test );
	
	if debug>1:
		pl.figure();
		sampleImage = np.reshape(images_training[:,0], [28, 28]);
		pl.imshow(sampleImage, cmap='gray');
		pl.show();
	
	if debug:
		print 'Number of training samples: ', nSamples_training
		print 'Number of test samples: ', nSamples_test
	
	dimLayers = [inputDim, hiddenDim, nClasses];
	
	smnn = SMNN(dimLayers, lambd, activation_fun, debug);
	
	success = smnn.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(smnn.predict(images_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == np.argmax(smnn.predict(images_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	