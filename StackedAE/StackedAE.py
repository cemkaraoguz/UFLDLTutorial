''' StackedAE.py
	
	Implementation of Sparse Autoencoder
	
	Author: Cem Karaoguz
	Date: 13.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize
import scipy.io
import scipy.linalg

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization
from UFL.SparseAutoencoder import SparseAutoencoder
from UFL.Softmax import Softmax

STACKEDAE_HIDDENLAYER_TYPES = ['sparseae']

class StackedAutoencoder:
	''' 
	Stacked Autoencoder
	'''

	def __init__(self,
	             inputDim,
				 outputDim,
				 hiddenLayerParams,
				 lambd=1e-4,
				 beta=3,
				 sparsityParam=0.1,
				 doFineTuning=True,
				 debug=0):
		''' 
		Initialization function of the Stacked Autoencoder class
		
		Arguments		
		inputDim			: Dimension of the input layer
		outputDim			: Dimension of the output layer
		hiddenLayerParams	: Parameters of hidden layers
		lambd				: weight decay parameter, default is 0.003
		beta				: weight of sparsity penalty term, default is 3
		sparsityParam		: weight of the sparsity in the cost function, default is 0.1
		doFineTuning		: Do fine tuning optimization step?
		debug				: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.outputDim = outputDim;
		self.lambd = lambd;
		self.beta = beta;
		self.sparsityParam = sparsityParam;
		self.doFineTuning = doFineTuning;
		self.hiddenLayerParams = hiddenLayerParams;
		
		assert self.inputDim>0, 'ERROR:StackedAutoencoder:init: Input layer dimension must be >0'
		assert self.outputDim>0, 'ERROR:StackedAutoencoder:init: Output layer dimension must be >0'
		
		# Check necessary keys inside the hidden layer parameters
		for p in hiddenLayerParams:
			assert ('id' in p.keys()), 'ERROR:StackedAutoencoder:init: ID field is mandatory for hidden layer parameter'
			assert ('featureDim' in p.keys()), 'ERROR:StackedAutoencoder:init: featureDim field is mandatory for hidden layer parameter'
			if not (p['id'] in STACKEDAE_HIDDENLAYER_TYPES):
				print 'ERROR:StackedAutoencoder:init:', p['id'], 'is not allowed. Valid ids are:'
				print STACKEDAE_HIDDENLAYER_TYPES
				sys.exit();
			
		self.nHiddenLayers = len(hiddenLayerParams);
		self.nLayers = 1 + self.nHiddenLayers + 1;
		
		assert self.nHiddenLayers==2, 'ERROR:StackedAutoencoder:init: The number of hidden layers should be 2'
		
		# For analyzing different network states
		self.doTrainLayer = [1, 1, 1];
		
		self.layerDims = []
		self.layerDims.append(self.inputDim);
		for i in range(self.nHiddenLayers):
			self.layerDims.append(hiddenLayerParams[i]['featureDim'])
		self.layerDims.append(self.outputDim);
		
		# Initialize hidden layers
		self.hiddenLayers = []
		for layer in range(self.nHiddenLayers):
			layerparams = hiddenLayerParams[layer];
			if layerparams['id']=='sparseae':
				dimLayers = [self.layerDims[layer], self.layerDims[layer+1], self.layerDims[layer]];
				sae = SparseAutoencoder.SparseAutoencoder(dimLayers=dimLayers, 
														  lambda_w=layerparams['lambd'], 
														  beta=layerparams['beta'], 
														  sparsityParam=layerparams['sparsityParam'], 
														  actFunctions=layerparams['actFunctions'], 
														  debug=layerparams['debug']);
				self.hiddenLayers.append(sae);
				
			else:
				print 'ERROR:StackedAutoencoder:init: identity ', layerparams['id'], ' is not recognized for layer:', layer;
				sys.exit();
				
		# Initialize output layer: softmax
		self.softmaxmodel = Softmax.Softmax(self.layerDims[-2], self.layerDims[-1], self.debug-1);
		
		if debug:
			print 'DEBUG:StackedAutoencoder:init: initialized for layer dimensions: ', self.layerDims;
			print 'DEBUG:StackedAutoencoder:init: initialized for lambd: ', self.lambd;
			print 'DEBUG:StackedAutoencoder:init: initialized for beta: ', self.beta;
			print 'DEBUG:StackedAutoencoder:init: initialized for sparsityParam: ', self.sparsityParam;
			print 'DEBUG:StackedAutoencoder:init: initialized for doFineTuning?: ', self.doFineTuning;
			print
		
		self.isInitialized = True;
		
	def getNetworkParametersStacked(self):
		''' 
		Returns the parameters of the network in a stacked form
		'''
		weights = [];
		biases = [];
		for layer in range(self.nHiddenLayers):
			W = self.hiddenLayers[layer].getWeights_asLayer();
			b = self.hiddenLayers[layer].getBiases_asLayer();
			weights.append(W);
			biases.append(b);
		
		W = self.softmaxmodel.getWeights()
		weights.append(W);
			
		return weights, biases
	
	def getNetworkParametersLinear(self):
		''' 
		Returns the parameters of the network in linear form
		'''
		params = [];
		for layer in range(self.nHiddenLayers):
			W = self.hiddenLayers[layer].getWeights_asLayer();
			b = self.hiddenLayers[layer].getBiases_asLayer();
			params = np.hstack((params, W.flatten(), b.flatten()));
		
		W = self.softmaxmodel.getWeights()
		params = np.hstack((params, W.flatten()));
		
		return params
			
	def setNetworkParameters(self, params):
		''' 
		Updates the internal Stacked Autoencoder parameters with the given ones
		
		Arguments
		params	: rolled parameters to set for the first layer of the network
		'''
		[weights, biases] = self.unrollParameters(params);
		
		for layer in range(self.nHiddenLayers):
			W = weights[layer];
			b = biases[layer];
			#self.hiddenLayers[layer].setParams(W, b);
			self.hiddenLayers[layer].setWeights_asLayer(W);
			self.hiddenLayers[layer].setBiases_asLayer(b);
			
		W = weights[-1];
		self.softmaxmodel.setWeights(W)
		
	def rollParameters(self, weights, biases):
		''' 
		Converts the parameters in matrix form into vector
		
		Arguments
		weights	: list of weight matrices of each layer 
		biases	: list of bias vectors of each layer 
		
		Returns
		params	: parameter vector
		'''
		nLayers = len(weights);
		params = np.array([]);
		for i in range(nLayers):
			
			if i<len(biases):
				assert weights[i].shape[0] == biases[i].shape[0], 'ERROR:StackedAutoencoder:rollParameters: weight and bias dimension mismatch '
			
				params = np.hstack((params, weights[i].flatten(), biases[i].flatten()))
				
			else:
				params = np.hstack((params, weights[i].flatten()))
		
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
		weights = [];
		biases = [];
		read_start = 0;
		read_end = 0;
		
		for i in range(self.nHiddenLayers):
			# set the end index for read
			read_end = read_start + self.layerDims[i+1]*self.layerDims[i];
			# read the weights for the current layer
			w = params[read_start:read_end];
			# reshape and the weights
			weights.append( np.reshape(w, (self.layerDims[i+1], self.layerDims[i])) );
			# set the start index for the next read
			read_start = read_end;
			# set the end index for the next read
			read_end = read_start + self.layerDims[i+1];
			# read the bias terms
			b = params[read_start:read_end];
			# reshape and store the bias
			biases.append( np.reshape(b, (self.layerDims[i+1], 1)) )
			# set the start index for the next read
			read_start = read_end;
		
		# Softmax layer
		read_end = read_start + self.layerDims[-1]*self.layerDims[-2];
		w = params[read_start:read_end];
		weights.append( np.reshape(w, (self.layerDims[-1], self.layerDims[-2])) );
		
		return weights, biases;
		
	def doForwardPropagation(self, X, weights, biases):
		''' 
		Computes the forward propagation of the input in the network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		weights		: list of weight matrices of each layer
		biases		: list of bias vectors of each layer
		
		Returns
		activities	: list of activation matrices (h) of each layer (output of neuron after activation function)
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:doForwardPropagation: The instance is not properly initialized'
		
		activities = [];
		indata = X;
		for i in range(len(self.hiddenLayers)):
		
			# Get original weights and biases of the sparse autoencoder layers
			W_sa = self.hiddenLayers[i].getWeights();
			b_sa = self.hiddenLayers[i].getBiases();
			# Replace the first layer's weights and biases with the current ones
			W_sa[0] = weights[i];
			b_sa[0] = biases[i];
			# Do the forward prop. with the new weights
			[outputs_sa, activities_sa] = self.hiddenLayers[i].doForwardPropagation(indata, W_sa, b_sa);
			# Get the activity of the first layer
			activity = activities_sa[0]
			activities.append(activity);
			
			indata = activity;
		
		outdata = self.softmaxmodel.doForwardPropagation(indata, weights[-1]);
		#outdata = np.dot(weights[-1], indata);
		
		# Convert output to probabilities:
		aux2 = AuxFunctions.doUnbalancedMatrixOperation(outdata, np.amax(outdata, 0), 'sub', axis=0); #Substracts the maximm value of the matrix "aux".
		aux3 = np.exp(aux2);
		y = AuxFunctions.doUnbalancedMatrixOperation(aux3, np.sum(aux3, 0), 'div', axis=0); #I divides the vector "aux3" by the sum of its elements.

		activities.append(y);
		
		return activities;
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the Stacked Autoencoder objective function for given 
		features (theta), data matrix (X) and corresponding labels (y):
		
		1/nSamples * sum(log(nonzero(Y * H_out))) + 1/2*lambda*sum(W_s)
		
		where Y is ground truth matrix, a binary matrix where for each column 
		(i.e. sample) the row corresponding to the true class is one and the rest is zero,
		H_out is the activity matrix of the output layer,
		W_s is the weights of the softmax layer.
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:computeCost: The instance is not properly initialized'
		
		f = 0;
		
		nSamples = X.shape[1];
		
		[weights, biases] = self.unrollParameters(theta);
		
		activities = self.doForwardPropagation(X, weights, biases);

		aux1 = np.repeat(np.reshape(range(self.outputDim), [1, self.outputDim]), nSamples, 0)
		aux2 = np.repeat(np.reshape(y, [nSamples, 1]), self.outputDim, 1);
		groundTruth = np.transpose((aux1==aux2).astype(int));
		
		aux3 = groundTruth * activities[-1];
		aux4 = np.log(aux3[aux3 != 0]); # Extract non-zero entries.
		
		cost_fidelity = -np.mean(aux4, 0);
		cost_regularization = (self.lambd/2.0) * np.sum(weights[-1]**2);
		
		f = cost_fidelity + cost_regularization;
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the Stacked Autoencoder objective function wrt parameters
		(theta) for a given data matrix (X) and corresponding labels (y) with error
		back propagation:
		
		E_out = Y - H_out
		E_{l-1} = W_{l-1}' * E_{l} * df(Z{l-1})/dz
		
		where Y is ground truth matrix, a binary matrix where for each column (i.e. sample) 
		the row corresponding to the true class is one and the rest is zero,
		H_out is the activity matrix of the output layer,
		df(Z)/dz is the derivatives of the activation function at points Z which is
		
		df(Z)/dz = f(Z)*(1-f(Z)) 	for sigmoid activation function and
		df(Z)/dz = Z 				for identity activation function. 
		
		Gradients are then, computed as:
		
		dJ(W,b;X,y)/dW_{l} = E_{l+1} * H_{l}'
		dJ(W,b;X,y)/db_{l} = sum(E_{l+1})
		
		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (number of parameters*number of classes,)
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:computeGradient: The instance is not properly initialized'
		
		nSamples = X.shape[1];
		
		[weights, biases] = self.unrollParameters(theta);
		
		activities = self.doForwardPropagation(X, weights, biases);
		
		aux1 = np.repeat(np.reshape(range(self.outputDim), [1, self.outputDim]), nSamples, 0)
		aux2 = np.repeat(np.reshape(y, [nSamples, 1]), self.outputDim, 1);
		groundTruth = np.transpose((aux1==aux2).astype(int));
		
		error_y_fidelity = (-1.0/nSamples)*(groundTruth - activities[-1]);
		error_y_regularization = self.lambd * weights[-1];
		
		deltas = []
		error = error_y_fidelity;
		for i in range(self.nHiddenLayers):
			error_prop = ( (np.dot(np.transpose(weights[-(i+1)]), error)));
			delta_prop = error_prop * (activities[-(i+2)]*(1 - activities[-(i+2)]));
			deltas.append(delta_prop);
			error = delta_prop;
		
		deltas = list(reversed(deltas));
		
		gradients_W = [];
		gradients_b = [];
		for layer in range(self.nHiddenLayers):
			if layer==0:
				x_in = X;
			else:
				x_in = activities[layer-1];
			
			gradients_W.append( np.dot(deltas[layer], np.transpose(x_in)) );
			gradients_b.append( np.sum(deltas[layer], 1) );
		
		softmaxThetaGrad = np.transpose(np.dot(activities[-2], np.transpose(error_y_fidelity))) + error_y_regularization;
		
		gradients_W.append(softmaxThetaGrad);
		
		return self.rollParameters(gradients_W, gradients_b);
		
	def testGradient(self, X, y):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [input dim., number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:testGradient: The instance is not properly initialized'
		
		if self.debug: print 'DEBUG:StackedAutoencoder:testGradient: Testing gradient computation...'
		
		result = 0;
		
		params = self.getNetworkParametersLinear();
		
		grad = self.computeGradient(params, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=params, args=((X, y)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:StackedAutoencoder:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:StackedAutoencoder:testGradient:Gradient check PASSED!'
				print
			
			result = 0;
			
		else:
			if self.debug:
				print 'DEBUG:StackedAutoencoder:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:StackedAutoencoder:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
			
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the Stacked Autoencoder model
		
		Optimization is done in two steps:
		1) Optimization of each layers individually
		2) Fine tuning of the whole network
		
		In Step 1, each layer is trained individually. Autoencoder layers are trained 
		as normal 3-layer Autoencoder networks. After training only the weights of 
		the first layer are used in the Stacked Autoencoder Network. Outputs of this 
		layer become input for the next Stacked Autoencoder	layer which could be 
		another Autoencoder or a Softmax layer. Softmax layer is also trained individually, 
		inputs being the outputs of the last Autoencoder layer.
		
		In Step 2, the whole network is fine tuned via error back-propagation.
		
		Arguments
		X		: data in the form [input dim., number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:StackedAutoencoder:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print "DEBUG:StackedAutoencoder:optimizeParameters: Optimizing parameters..."
		
		result = 0;
		
		indata = X;
		for i in range(self.nHiddenLayers):
			
			if self.doTrainLayer[i]:
				if debug: print "DEBUG:StackedAutoencoder:optimizeParameters: Training hidden layer ", i+1, ': ', self.hiddenLayerParams[i]['id']
				
				# Optimize the cost function
				result = self.hiddenLayers[i].optimizeParameters(indata);
			
			# Set the output of the current autoencoder as the input of the next layer
			indata = self.hiddenLayers[i].predict(indata);
		
		if self.doTrainLayer[-1]:
			if debug: print "DEBUG:StackedAutoencoder:optimizeParameters: Training output layer : Softmax"
			
			result = self.softmaxmodel.optimizeParameters(indata, y);
		
		# Get new network parameters
		params = self.getNetworkParametersLinear();
		
		if self.doFineTuning:
			if debug: print "DEBUG:StackedAutoencoder:optimizeParameters: Fine tuning the deep network..."
			
			# Set optimization options
			method = 'L-BFGS-B'
			options = {};
			options['maxiter'] = 100;
			
			if self.debug:
				options['disp'] = True;
			else:
				options['disp'] = False;
			
			# Optimize the cost function
			result_finetune = minimize(fun=self.computeCost, jac=self.computeGradient, x0=params, args=(X,y), method=method, options=options)
			
			result = result_finetune.success;
			
			if self.debug:
				print 'DEBUG:StackedAutoencoder:optimizeParameters:',  result_finetune.message
			
			# Set the new values
			self.setNetworkParameters(result_finetune.x);
			
		return result;

	def predict(self, X):
		'''
		Applies the Stacked Autoencoder model to the given data
		
		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		output of the network in the form [output dim., number of samples]
		'''
		assert self.isInitialized, 'ERROR:StackedAutoencoder:predict: The instance is not properly initialized'
		
		[weights, biases] = self.getNetworkParametersStacked();
		
		activities = self.doForwardPropagation(X, weights, biases);

		return activities[-1];
	
	
if __name__ == '__main__':
	
	# Test gradient computation?
	doTest = True;
	
	if (doTest):
		debug 			= 2;
		numPatches 		= 10;
		patchWidth 		= 2;
		patchHeight 	= 2;
		imChannels		= 1;
		inputDim 		= patchWidth * patchHeight * imChannels;
		outputDim		= 2;

		# Parameters of Autoencoders
		params_ae					= {}
		params_ae['id']				= 'sparseae'
		params_ae['lambd']		 	= 1e-4;
		params_ae['beta']			= 3;
		params_ae['sparsityParam']	= 0.1;
		params_ae['featureDim']		= 5;
		params_ae['actFunctions']	= [SparseAutoencoder.ACTIVATION_FUNCTION_SIGMOID, SparseAutoencoder.ACTIVATION_FUNCTION_SIGMOID]
		params_ae['debug']			= 1;
		
		# Both layers are identical
		hiddenLayerParams = [params_ae, params_ae];
		
		#testdata = np.random.rand(inputDim, numPatches);
		#testlabel = np.round(np.random.rand(numPatches));
		testlabel = np.array([1,1,1,1,1,2,2,2,2,2])-1;
		testdata = np.transpose(np.reshape(range(inputDim*numPatches), [numPatches,inputDim]));
		
		StackedAE_test = StackedAutoencoder( inputDim, outputDim, hiddenLayerParams, debug=debug);
		
		print 'Checking gradient...'
		
		StackedAE_test.testGradient(testdata, testlabel);

	if 1:
	  mnist_lbl_filename_training = '/home/cem/develop/UFL/data/train-labels-idx1-ubyte';
	  mnist_img_filename_training = '/home/cem/develop/UFL/data/train-images-idx3-ubyte';
	  mnist_lbl_filename_test = '/home/cem/develop/UFL/data/t10k-labels-idx1-ubyte';
	  mnist_img_filename_test = '/home/cem/develop/UFL/data/t10k-images-idx3-ubyte';
	else:
	  mnist_lbl_filename_training = 'C://develop//python//UFL//data//train-labels-idx1-ubyte';
	  mnist_img_filename_training = 'C://develop//python//UFL//data//train-images-idx3-ubyte';
	  mnist_lbl_filename_test 	= 'C://develop//python//UFL//data//t10k-labels-idx1-ubyte';
	  mnist_img_filename_test 	= 'C://develop//python//UFL//data//t10k-images-idx3-ubyte';
	  
	debug 				= 1;
	nSamples_max_train 	= 20000;
	nSamples_max_test 	= 30000;
	imWidth				= 28;
	imHeight			= 28;
	imChannels			= 1;
	inputDim 			= imWidth * imHeight * imChannels;
	outputDim			= 10;
	lambd 				= 1e-4;
	beta 				= 3;
	sparsityParam 		= 0.1;
	doFineTuning		= True;
	
	# Parameters of Autoencoders
	params_ae					= {}
	params_ae['id']				= 'sparseae'
	params_ae['lambd']		 	= 3e-3;
	params_ae['beta']			= 3;
	params_ae['sparsityParam']	= 0.1;
	params_ae['featureDim']		= 200;
	params_ae['actFunctions']	= [SparseAutoencoder.ACTIVATION_FUNCTION_SIGMOID, SparseAutoencoder.ACTIVATION_FUNCTION_SIGMOID]
	params_ae['debug']			= 1;
	
	# Both layers are identical
	hiddenLayerParams		= [params_ae, params_ae];
	
	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_max_train);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_max_train);
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_max_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_max_test);

	# Normalize data 
	images_training = images_training / 255.0;
	images_test = images_test / 255.0;
	
	StackedAE = StackedAutoencoder( inputDim,
									outputDim,
									hiddenLayerParams,
									lambd=lambd,
									beta=beta,
									sparsityParam=sparsityParam,
									doFineTuning=doFineTuning,
									debug=debug);
	
	success = StackedAE.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(StackedAE.predict(images_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == np.argmax(StackedAE.predict(images_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	
	# Visualize the learned bases
	[weights, biases] = StackedAE.getNetworkParametersStacked();
	Visualization.displayNetwork(np.transpose(weights[0]));
	