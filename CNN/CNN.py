''' CNN.py	
		
	Implementation of Convolutional Neural Network
	
	Author: Cem Karaoguz
	Date: 19.03.2015
	Version: 1.0
	
	TODO: implement max pooling
	TODO: deconvolution
	TODO: Try multiple layers
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize
import scipy.io
import scipy.linalg
import scipy.signal

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization
from UFL.Softmax import Softmax

INDEX_X 				= 0
INDEX_Y 				= 1
INDEX_ACTIVATION_CONV 	= 0
INDEX_ACTIVATION_POOL 	= 1
POOLING_MEAN			= 0;
POOLING_MAX				= 1;
CNN_POOLING_FUNCTIONS	= [POOLING_MEAN, POOLING_MAX]

class ConvLayer:
	''' 
	Convolutional Neural Network Layer
	
	May consist of
	- Convolution
	- Non-linear filtering 
	- Pooling
	'''
	
	def __init__(self, inputDim, numFilters, filterDim, poolDim, poolingFunction, debug=0):
		''' 
		Initialization function of the Convolutional Neural Network Layer class
		
		Arguments
		inputDim			: Dimension of the input layer
		filterDim			: Filter size for convolution layer
		numFilters			: Number of filters for convolution layer
		poolDim				: Pooling dimension, (should divide imageDim-filterDim+1)
		poolingFunction		: Pooling function, [POOLING_MEAN]
		debug				: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.filterDim = filterDim;
		self.numFilters = numFilters;
		self.poolDim = poolDim;
		self.poolingFunction = poolingFunction;

		assert len(self.inputDim)==2, 'ERROR:ConvLayer:init: input layer dimension must be two dimensional vector'
		assert self.inputDim[INDEX_X]>0, 'ERROR:ConvLayer:init: input layer dimensions must be >0'
		assert self.inputDim[INDEX_Y]>0, 'ERROR:ConvLayer:init: input layer dimensions must be >0'
		assert len(self.filterDim)==2, 'ERROR:ConvLayer:init: filter layer dimension must be two dimensional vector'
		assert self.filterDim[INDEX_X]>0, 'ERROR:ConvLayer:init: filter layer dimensions must be >0'
		assert self.filterDim[INDEX_Y]>0, 'ERROR:ConvLayer:init: filter layer dimensions must be >0'
		assert len(self.poolDim)==2, 'ERROR:ConvLayer:init: pooling layer dimension must be two dimensional vector'
		assert self.poolDim[INDEX_X]>0, 'ERROR:ConvLayer:init: pooling layer dimensions must be >0'
		assert self.poolDim[INDEX_Y]>0, 'ERROR:ConvLayer:init: pooling layer dimensions must be >0'
		assert self.poolingFunction in CNN_POOLING_FUNCTIONS, 'ERROR:CNN:Init: Pooling function not recognized'
		
		# Set layer topology
		self.weights = 1e-1 * np.random.randn(self.filterDim[INDEX_X], self.filterDim[INDEX_Y], self.numFilters);
		self.biases = np.zeros([self.numFilters, 1]);
		
		self.weightPrototype = (self.filterDim[INDEX_X], self.filterDim[INDEX_Y], self.numFilters)
		self.biasPrototype = (self.numFilters, 1)
		
		# Only for testing
		if 0:
			tmp = scipy.io.loadmat('cnn_weights_DEBUG.mat');
			self.weights = tmp['Wc']
			self.biases = tmp['bc']

		
		# Dimension of convolved image
		self.convDim = [0, 0];
		self.convDim[INDEX_X] = self.inputDim[INDEX_X] - self.filterDim[INDEX_X] + 1;
		self.convDim[INDEX_Y] = self.inputDim[INDEX_Y] - self.filterDim[INDEX_Y] + 1;

		assert np.mod(self.convDim[INDEX_X], self.poolDim[INDEX_X])==0, 'poolDim must divide imageDim - filterDim + 1';
		assert np.mod(self.convDim[INDEX_Y], self.poolDim[INDEX_Y])==0, 'poolDim must divide imageDim - filterDim + 1';
		
		# Dimension of pooling layer
		self.outputDim = [0, 0];
		self.outputDim[INDEX_X] = self.convDim[INDEX_X]/self.poolDim[INDEX_X];
		self.outputDim[INDEX_Y] = self.convDim[INDEX_Y]/self.poolDim[INDEX_Y];
		
		if debug:
			print 'DEBUG:ConvLayer:init: initialized for inputDim: ', self.inputDim;
			print 'DEBUG:ConvLayer:init: initialized for filterDim: ', self.filterDim;
			print 'DEBUG:ConvLayer:init: initialized for numFilters: ', self.numFilters;
			print 'DEBUG:ConvLayer:init: initialized for convDim: ', self.convDim;
			print 'DEBUG:ConvLayer:init: initialized for poolDim: ', self.poolDim;
			print 'DEBUG:ConvLayer:init: initialized for outputDim: ', self.outputDim;			
			print
			
		self.isInitialized = True;
		
	def doForwardPropagation(self, X, weights, biases):
		''' 
		Computes the forward propagation of the input in the network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		weights		: list of weight matrices of each layer
		biases		: list of bias vectors of each layer
		
		Returns
		activities	: list of activation matrices from convolution and pooling layers, respectively
		'''
		assert self.isInitialized, 'ERROR:ConvLayer:doForwardPropagation: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters([weights], [self.weightPrototype]), 'ERROR:ConvLayer:doForwardPropagation: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters([biases], [self.biasPrototype]), 'ERROR:ConvLayer:doForwardPropagation: bias dimension does not match the network topology';
		
		# Convolution
		activations_conv = convolve(self.filterDim, self.numFilters, X, weights, biases);
		# Pooling
		activations_pool = pool(self.poolDim, activations_conv, self.poolingFunction);
			
		return [activations_conv, activations_pool];
	
	def backPropagateError(self, error, layer_in, layer_out, weights):
		''' 
		Computes the back propagation of the error in the layer:
		
		E_{in} = upsample(W_{in} * E_{out}) * df(Z{out})/dz
		
		where E_{out} is the error matrix for the output of the layer (error),
		f(Z{out}) is the output activity of the layer (layer_out),
		df(Z)/dz is the derivatives of the activation function at points Z,
		E_{in} is the propagated error matrix.
		
		The gradients are computed via convolution:
		
		dJ(W,b;X,y)/dW_{l-1} = conv(E_{in}, H_{in})
		dJ(W,b;X,y)/db_{l-1} = sum(E_{in})
		
		Arguments
		error			: error matrix of the output layer (i.e. pooling sub-layer) with columns corresponding to the samples, rows corresponding to the units
		layer_in		: data given to the network layer
		layer_out		: output of the network layer
		
		Returns
		error_upsampled	: back-propagated error
		Wc_grad			: weight gradients
		bc_grad			: bias gradients
		'''
		assert self.isInitialized, 'ERROR:ConvLayer:backPropagateError: The instance is not properly initialized'
	
		Wc_grad = np.zeros(self.weights.shape);
		bc_grad = np.zeros(self.biases.shape);
		
		numData = layer_in.shape[2];
		
		error_upsampled = np.zeros([self.convDim[INDEX_X], self.convDim[INDEX_Y], self.numFilters, numData]);

		for i in range(numData):
			for filterNum in range(self.numFilters):
				# Upsample the incoming error using kron
				if self.poolingFunction==POOLING_MEAN:
					aux1 = (1.0/(self.poolDim[INDEX_X]*self.poolDim[INDEX_Y])) * np.kron(error[:,:,filterNum,i], np.ones([self.poolDim[INDEX_X], self.poolDim[INDEX_Y]]));
					error_upsampled[:, :, filterNum, i] = aux1 * layer_out[INDEX_ACTIVATION_CONV][:,:,filterNum,i] * (1 - layer_out[INDEX_ACTIVATION_CONV][:,:,filterNum,i]);
				else:
					assert 0, 'ERROR:ConvLayer:backPropagateError: Pooling function not recognized'
				
				# Convolution:
				aux2 = error_upsampled[:,:,filterNum,i];
				aux2 = np.rot90(aux2, 2);
				aux3 = scipy.signal.convolve2d(layer_in[:,:,i], aux2, 'valid');
				
				Wc_grad[:,:,filterNum] = Wc_grad[:,:,filterNum] + aux3;
				bc_grad[filterNum] = bc_grad[filterNum] + np.sum(aux2);
				
		return error_upsampled, Wc_grad, bc_grad
		
	def getParameters(self):
		'''
		Returns weights and biases of the layer
		'''
		return self.weights, self.biases
		
	def setParameters(self, W, b):
		'''
		Sets the weights and biases of the layer with the given parameters
		
		Arguments
		W	: weights to set
		b	: biases to set
		'''
		assert AuxFunctions.checkNetworkParameters([W], [self.weightPrototype]), 'ERROR:ConvLayer:setParameters: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters([b], [self.biasPrototype]), 'ERROR:ConvLayer:setParameters: bias dimension does not match the network topology';

		self.weights = W;
		self.biases = b;
		

def convolve(filterDim, numFilters, X, W, b):
	'''
	Returns the convolution of the features given by W and b with the given data X
	
	Arguments
	filterDim			: filter (feature) dimension
	numFilters			: number of feature maps
	X					: input data in the form images(r, c, image number)
	W					: weights i.e. features, is of shape (filterDim,filterDim,numFilters)
	b					: biases, is of shape (numFilters,1)
	
	Returns
	convolvedFeatures	: matrix of convolved features in the form convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
	'''
	inputDimX = X.shape[INDEX_X];
	inputDimY = X.shape[INDEX_Y];
	numData   = X.shape[2];
	
	convDimX = inputDimX - filterDim[INDEX_X] + 1;
	convDimY = inputDimY - filterDim[INDEX_Y] + 1;

	convolvedFeatures = np.zeros([convDimX, convDimY, numFilters, numData]);

	for i in range(numData):
	  for filterNum in range (numFilters):

		# Convolution of image with feature matrix
		convolvedImage = np.zeros([convDimX, convDimY]);

		# Obtain the feature (filterDim x filterDim) needed during the convolution
		filter = W[:,:,filterNum];
		
		# Flip the feature matrix because of the definition of convolution, as explained later
		filter = np.rot90(filter, 2);
		  
		# Obtain data
		data = X[:,:,i];

		#Convolve "filter" with "data", adding the result to convolvedImage
		convolvedImage = scipy.signal.convolve2d(data, filter, mode='valid');
		
		# Add the bias unit
		# Then, apply the sigmoid function to get the hidden activation
		convolvedImage = AuxFunctions.sigmoid(convolvedImage + b[filterNum]);
		
		convolvedFeatures[:,:,filterNum,i] = convolvedImage;
		
	return convolvedFeatures

def pool(poolDim, convolvedFeatures, poolingFunction):
	'''
	Pools the given convolved features
	
	Parameters:
	poolDim - dimension of pooling region
	convolvedFeatures - convolved features to pool (as given by cnnConvolve)
	                    convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
	
	Returns:
	pooledFeatures - matrix of pooled features in the form
	                 pooledFeatures(poolRow, poolCol, featureNum, imageNum)
	'''

	convolvedDimX = convolvedFeatures.shape[INDEX_X];
	convolvedDimY = convolvedFeatures.shape[INDEX_Y];
	numData   = convolvedFeatures.shape[3];
	numFilters = convolvedFeatures.shape[2];

	pooledFeatures = np.zeros([convolvedDimX/poolDim[INDEX_X], convolvedDimY/poolDim[INDEX_Y], numFilters, numData]);

	# Pool the convolved features in regions of poolDim x poolDim, to obtain the 
	# (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numData 
	# matrix pooledFeatures, such that pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
	# value of the featureNum feature for the imageNum image pooled over the corresponding (poolRow, poolCol) pooling region. 

	if poolingFunction==POOLING_MEAN:
		poolFilter = np.ones([poolDim[INDEX_X], poolDim[INDEX_Y]]) / (1.0 * poolDim[INDEX_X] * poolDim[INDEX_Y]);
	elif poolingFunction==POOLING_MAX:
		# not yet implemented
		poolFilter = np.ones([poolDim[INDEX_X], poolDim[INDEX_Y]]) / (1.0 * poolDim[INDEX_X] * poolDim[INDEX_Y]);
	else:
		assert 0, 'ERROR:pool: Pooling function not recognized'
		
	#poolFilter = np.rot90(poolFilter, 2);
	
	for i in range(numData):
	  for filterNum in range(numFilters):
		  pooledFeature = scipy.signal.convolve2d(convolvedFeatures[:, :, filterNum, i], poolFilter, 'valid');
		  pooledFeatures[:, :, filterNum, i] = pooledFeature[0:convolvedDimX-poolDim[INDEX_X]+1:poolDim[INDEX_X], 0:convolvedDimY-poolDim[INDEX_Y]+1:poolDim[INDEX_Y]];
	
	return pooledFeatures
	
	
class CNN:
	''' 
	Convolutional Neural Network
	'''

	def __init__(self,
	             inputDim,
				 outputDim,
				 layerParams,
				 epochs = 3,
				 minibatch = 256,
				 alpha = 1e-1,
				 momentum = 0.95,
				 debug=0):
		''' 
		Initialization function of the Convolutional Neural Network class
		
		Arguments
		inputDim			: Dimension of the input layer
		outputDim			: Dimension of the output layer
		layerParams			: Parameters for CNN Layers
		epochs				: Training stocahastic gradient descent (SGD) epochs, default is 3
		minibatch			: Number of samples to use in each SGD epoch, default is 256
		alpha				: Initial learning rate, default is 1e-1
		momentum			: Momentum constant, default is 0.95
		debug				: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.outputDim = outputDim;
		self.layerParams = layerParams;
		self.epochs = epochs;
		self.minibatch = minibatch;
		self.alpha = alpha;
		self.momentum = momentum;
		
		assert len(self.inputDim)==2, 'ERROR:CNN:init: Input layer dimension must be two dimensional vector'
		assert self.inputDim[0]>0, 'ERROR:CNN:init: Input layer dimensions must be >0'
		assert self.inputDim[1]>0, 'ERROR:CNN:init: Input layer dimensions must be >0'
		assert self.outputDim>0, 'ERROR:CNN:init: Output layer dimension must be >0'
		assert self.epochs>0, 'ERROR:CNN:init: epochs must be >0'
		assert self.minibatch>0, 'ERROR:CNN:init: minibatch size must be >0'
		assert self.alpha>0, 'ERROR:CNN:init: alpha must be >0'
		assert self.momentum>0, 'ERROR:CNN:init: momentum must be >0'
		
		# Initialize layers
		inputDimLayer = inputDim;
		self.weightPrototypes = [];
		self.biasPrototypes = [];
		self.layers = [];
		for i in range(len(self.layerParams)):
			layer = ConvLayer( inputDimLayer,
                               layerParams[i]['numFilters'],
						       layerParams[i]['filterDim'],
							   layerParams[i]['poolDim'],
							   layerParams[i]['poolingFunction'],
							   layerParams[i]['debug']);
			
			self.layers.append(layer);
			self.weightPrototypes.append(layer.weightPrototype)
			self.biasPrototypes.append(layer.biasPrototype)
			
			# Output dimension of the current layer is the input dimension for the next layer
			inputDimLayer = layer.outputDim;
		
		# Initialize output layer: softmax
		hiddenSize = layer.outputDim[INDEX_X] * layer.outputDim[INDEX_Y] * layer.numFilters
		r  = np.sqrt(6) / np.sqrt(self.outputDim + hiddenSize + 1);
		self.weights = np.random.rand(self.outputDim, hiddenSize) * 2 * r - r;
		self.biases = np.zeros([self.outputDim, 1]);
		
		self.weightPrototypes.append((self.outputDim, hiddenSize));
		self.biasPrototypes.append((self.outputDim, 1));
		
		if 0:
			# Only for testing
			tmp = scipy.io.loadmat('cnn_weights_DEBUG.mat');
			self.weights = tmp['Wd']
			self.biases = tmp['bd']
		
		if debug:
			print 'DEBUG:CNN:init: initialized for inputDim: ', self.inputDim;
			print 'DEBUG:CNN:init: initialized for outputDim: ', self.outputDim;
			print 'DEBUG:CNN:init: initialized for conv. layers: ', len(self.layers);
			print 'DEBUG:CNN:init: initialized for epochs: ', self.epochs;
			print 'DEBUG:CNN:init: initialized for minibatch: ', self.minibatch;
			print 'DEBUG:CNN:init: initialized for alpha: ', self.alpha;
			print 'DEBUG:CNN:init: initialized for momentum: ', self.momentum;
			print
		
		self.isInitialized = True;
		
	def getNetworkParameters(self):
		''' 
		Returns the parameters of the network in a stacked form
		'''
		weights = [];
		biases = [];
		for i in range(len(self.layers)):
			W, b = self.layers[i].getParameters();
			weights.append(W);
			biases.append(b);
		
		weights.append(self.weights);
		biases.append(self.biases);
		
		return weights, biases
	
	def setNetworkParameters(self, weights, biases):
		''' 
		Returns the parameters of the network in a stacked form
		
		Arguments
		weights	: list weights to set for each layer
		biases	: list of biases to set for each layer
		'''
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:CNN:setNetworkParameters: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:CNN:setNetworkParameters: bias dimension does not match the network topology';
		
		for i in range(len(self.layers)):
			W = weights[i];
			b = biases[i];
			self.layers[i].setParameters(W, b); # Size check is done in the layer
			
		self.weights = weights[-1];
		self.biases = biases[-1];
		
	def rollParameters(self, weights, biases):
		''' 
		Converts the parameters in matrix form into vector
		
		weights	: list of weight matrix of each layer 
		biases	: list of bias vector of each layer 
		'''
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:CNN:rollParameters: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:CNN:rollParameters: bias dimension does not match the network topology';
		
		params = np.array([]);
		for i in range(len(weights)):
			params = np.hstack((params, weights[i].flatten(), biases[i].flatten()))
			
		return params
		
	def unrollParameters(self, params):
		''' 
		Converts the vectorized parameters into matrix
		
		params: parameters to unroll
		'''
		weights = [];
		biases = [];
		read_start = 0;
		read_end = 0;
		
		# Convolutional layers
		for i in range(len(self.layers)):
			# set the end index for read
			read_end = read_start + self.layers[i].filterDim[INDEX_X]*self.layers[i].filterDim[INDEX_Y]*self.layers[i].numFilters;
			# read the weights for the current layer
			w = params[read_start:read_end];
			# reshape and the weights
			weights.append( np.reshape(w, (self.layers[i].filterDim[INDEX_X], self.layers[i].filterDim[INDEX_Y], self.layers[i].numFilters)) );
			# set the start index for the next read
			read_start = read_end;
			# set the end index for the next read
			read_end = read_start + self.layers[i].numFilters;
			# read the bias terms
			b = params[read_start:read_end];
			# reshape and store the bias
			biases.append( np.reshape(b, (self.layers[i].numFilters, 1)) )
			# set the start index for the next read
			read_start = read_end;
		
		# Softmax layer
		read_end = read_start+np.size(self.weights)
		w = params[read_start:read_end];
		weights.append( np.reshape(w, self.weights.shape) );
		# set the start index for the next read
		read_start = read_end;
		# set the end index for the next read
		read_end = read_start + len(self.biases);
		b = params[read_start:read_end];
		biases.append(np.reshape(b, self.biases.shape))
		
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:CNN:unrollParameters: dimensions of given parameters do not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:CNN:unrollParameters: dimensions of given parameters do not match the network topology';
		
		return weights, biases;
		
	def doForwardPropagation(self, X, weights, biases):
		''' 
		Computes the forward propagation of the input in the CNN.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		weights		: list of weight matrices of each layer
		biases		: list of bias vectors of each layer
		
		Returns
		activations	: list of activation matrices (h) of each layer (output of neuron after activation function)
		'''
		assert self.isInitialized, 'ERROR:CNN:doForwardPropagation: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:CNN:doForwardPropagation: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:CNN:doForwardPropagation: bias dimension does not match the network topology';
		
		activations = [];
		# Input to the network
		indata = X;
		# Propagate through the convolutional layers
		for i in range(len(self.layers)):
			
			# Compute the activity of the current layer
			outdata = self.layers[i].doForwardPropagation(indata, weights[i], biases[i]);
			
			# Save the activity of the current layer
			activations.append(outdata);
			
			# Set the activity of the current layer as the input to the next layer
			indata = outdata[INDEX_ACTIVATION_POOL];
		
		# Compute the activity of the softmax (output) layer
		# Reshape input for the softmax layer
		indata = np.reshape(indata, [indata.shape[0]*indata.shape[1]*indata.shape[2], indata.shape[3]]);
		
		# Compute the activity
		#outdata = self.softmaxmodel.predict(indata);
		
		z = np.dot(weights[-1], indata) + np.repeat(biases[-1], X.shape[2], 1);
		h = np.exp(z);
		y = AuxFunctions.doUnbalancedMatrixOperation(h, np.sum(h, 0), 'div', axis=0 );
		
		# Save the activity
		activations.append(y);
		
		return activations;
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the CNN objective function for given parameters
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
		assert self.isInitialized, 'ERROR:CNN:computeCost: The instance is not properly initialized'
		
		f = 0;
		
		nSamples = X.shape[2];
		
		[weights, biases] = self.unrollParameters(theta);
		
		activations = self.doForwardPropagation(X, weights, biases);
		
		P = AuxFunctions.doUnbalancedMatrixOperation(activations[-1], np.sum(activations[-1], 0), 'div', axis=0);
		aux3 = np.transpose(np.log(P));
		aux4 = np.repeat(np.reshape(range(self.outputDim), [1, self.outputDim]), nSamples, 0)
		aux5 = np.repeat(np.reshape(y, [nSamples, 1]), self.outputDim, 1);
		aux6 = aux4==aux5;
		
		f = (-1.0/nSamples) * np.sum(aux3 * aux6.astype(int));
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the CNN objective function for given parameters,	data and corresponding labels
		using the back propagation. First, the error of the output (Softmax) layer is computed:
		
		E_out = (Y - P(y|X;theta))
		
		where Y is ground truth matrix, a binary matrix where for each column (i.e. sample) 
		the row corresponding to the true class is one and the rest is zero
		
		P(Y|X;theta) = exp(theta'*X)/sum_j(exp(theta_j'*X)),	j = 1 to number of classes
		
		The output error is then back propagated to the convolutional layer:
		
		error_conv = W_out' * E_out
		
		And this error is further propagated within the convolutional layers. Gradients are computed:
		
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
		assert self.isInitialized, 'ERROR:CNN:computeGradient: The instance is not properly initialized'
		
		gradients_W = [];
		gradients_b = [];
		
		nSamples = X.shape[2];
		
		[weights, biases] = self.unrollParameters(theta);
		
		activations = self.doForwardPropagation(X, weights, biases);
		
		# Error of the output layer
		P = AuxFunctions.doUnbalancedMatrixOperation(activations[-1], np.sum(activations[-1], 0), 'div', axis=0);
		aux4 = np.repeat(np.reshape(range(self.outputDim), [1, self.outputDim]), nSamples, 0)
		aux5 = np.repeat(np.reshape(y, [nSamples, 1]), self.outputDim, 1);
		aux6 = aux4==aux5;
		
		error_out = (-1.0/nSamples) * (np.transpose(aux6.astype(int)) - P);
		
		# Gradient of the output layer
		act = activations[-2][INDEX_ACTIVATION_POOL]
		act = np.reshape(act, [act.shape[0]*act.shape[1]*act.shape[2], act.shape[3]])
		W_grad = np.dot(error_out, np.transpose(act));
		b_grad = np.dot(error_out, np.ones([nSamples, 1]));
		
		gradients_W.append(W_grad);
		gradients_b.append(b_grad);
		
		# Propagation of error_out to the last pooling layer
		error_pool = np.reshape( (np.dot(np.transpose(weights[-1]), error_out)), [self.layers[-1].outputDim[INDEX_X], self.layers[-1].outputDim[INDEX_Y], self.layers[-1].numFilters, nSamples]);
		
		# Back propagation of error through the layers
		error = error_pool
		for i in range(len(self.layers)):
			# Layer input
			if i==(len(self.layers)-1):
				layer_in = X;
			else:
				layer_in = activations[len(self.layers)-1-i-1][INDEX_ACTIVATION_POOL];
			
			# Layer output
			layer_out = activations[len(self.layers)-1-i]
			# Backpropagate error
			#[error_bp, W_grad, b_grad] = self.layers[len(self.layers)-1-i].backPropagateError(error, layer_in, layer_out);
			[error_bp, W_grad, b_grad] = self.layers[len(self.layers)-1-i].backPropagateError(error, layer_in, layer_out, weights[i]);
			
			# Save gradients
			gradients_W.append(W_grad);
			gradients_b.append(b_grad);
			
			# Set error for the next (previous) layer
			error = error_bp;
			
		# Reverse gradients
		gradients_W = list(reversed(gradients_W))
		gradients_b = list(reversed(gradients_b))
		
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
		assert self.isInitialized, 'ERROR:CNN:testGradient: The instance is not properly initialized'
		
		if self.debug: print 'DEBUG:CNN:testGradient: Testing gradient computation...'
		
		result = 0;
		
		[weights, biases] = self.getNetworkParameters();
		
		params = self.rollParameters(weights, biases);
		
		grad = self.computeGradient(params, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=params, args=((X, y)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:CNN:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:CNN:testGradient:Gradient check PASSED!'
				print
				
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:CNN:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:CNN:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
			
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the CNN model using Stochastic Gradient Descent (SGD)
		Mini batches of data are used to perform SGD. Parameter update is done in the following
		way:
		
		theta = theta - v
		
		Velocity v is defined as:
		
		v = gamma * v + alpha * delta_theta
		
		where gamma is the momentum (how many iterations the previous gradients are incorporated
		into the current update), alpha is the learning rate, delta_theta is the gradient vector.
		
		Arguments
		X		: data in the form [input dim., number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:CNN:optimizeParameters: The instance is not properly initialized'
		
		result = 0;
		
		nSamples = X.shape[2];
		
		[weights, biases] = self.getNetworkParameters();
		
		params = self.rollParameters(weights, biases);
		
		alpha = self.alpha;
		
		# Setup for momentum
		mom = 0.5;
		momIncrease = 20;
		velocity = np.zeros(np.size(params));

		if self.debug: print 'DEBUG:CNN:optimizeParameters: Start optimizing parameters...'
		
		# SGD loop
		it = 0;
		for e in range(self.epochs):
			
			# Randomly permute indices of data for quick minibatch sampling
			rp = np.random.permutation(nSamples);
			
			for s in range(0, nSamples-self.minibatch+1, self.minibatch):
				it = it + 1;
				
				# increase momentum after momIncrease iterations
				if (it == momIncrease):
					mom = self.momentum;
				
				# get next randomly selected minibatch
				mb_data = X[:, :, rp[s:s+self.minibatch-1]];
				mb_labels = y[rp[s:s+self.minibatch-1]];

				# evaluate the objective function on the next minibatch
				cost = self.computeCost(params, mb_data, mb_labels);
				grad = self.computeGradient(params, mb_data, mb_labels);
				
				# Add in the weighted velocity vector to the gradient evaluated 
				# above scaled by the learning rate.
				velocity = (mom * velocity) + (alpha * grad);
				
				# Update the current weights theta according to the SGD update rule
				params = params - velocity;
				
				if self.debug:
					print 'DEBUG:CNN:optimizeParameters: Epoch', e+1, ': Cost on iteration', it, 'is', cost
				
			# Aneal learning rate by factor of two after each epoch
			alpha = alpha/2.0;
		
		[weights, biases] = self.unrollParameters(params);
		
		self.setNetworkParameters(weights, biases);
		
		return result;

	def predict(self, X):
		'''
		Applies the CNN model to the given data
		
		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		Output activity matrix of the network
		'''
		assert self.isInitialized, 'ERROR:CNN:predict: The instance is not properly initialized'
		
		[weights, biases] = self.getNetworkParameters();
		
		activities = self.doForwardPropagation(X, weights, biases);

		return activities[-1];
		
if __name__ == '__main__':
	
	# Test gradient computation?
	doTest = True;
	
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
	  
	if (doTest):
		debug 			= 2;
		numPatches 		= 10;
		patchWidth 		= 28;
		patchHeight 	= 28;
		outputDim		= 10;
		
		params_layer1						= {}
		params_layer1['numFilters']			= 2;
		params_layer1['filterDim']			= [2, 2];
		params_layer1['poolDim']			= [3, 3];
		params_layer1['poolingFunction']	= POOLING_MEAN;
		params_layer1['debug']				= 1;
		
		params_layers = [params_layer1];
		
		#testlabel = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, numPatches);	
		testlabel = np.array([1,2,3,4,3,2,1,2,3,4])-1;
		testdata = DataInputOutput.loadMNISTImages(mnist_img_filename_training, numPatches);
		testdata = testdata / 255.0;
		testdata = np.reshape(testdata, [patchWidth, patchHeight, testdata.shape[1]]);
		
		ConvNet_test = CNN( [patchWidth, patchHeight], outputDim, params_layers, debug=debug);
		
		print 'Checking gradient...'
		
		ConvNet_test.testGradient(testdata, testlabel);
	
	debug 				= 1;
	nSamples_max_train 	= 10000;
	nSamples_max_test 	= 10000;
	imWidth				= 28;
	imHeight			= 28;
	outputDim			= 10;
	epochs				= 3;
	minibatch			= 256;
	alpha				= 1e-1;
	momentum			= 0.95;
	nVisSamples 		= 10;
	
	# Parameters for convolutional layers
	params_layer1						= {}
	params_layer1['numFilters']			= 20;
	params_layer1['filterDim']			= [9, 9];
	params_layer1['poolDim']			= [2, 2];
	params_layer1['poolingFunction']	= POOLING_MEAN;
	params_layer1['debug']				= 1;
	
	params_layers = [params_layer1];
	
	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_max_train);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_max_train);

	# Normalize data 
	images_training = images_training / 255.0;
	images_training = np.reshape(images_training, [patchWidth, patchHeight, images_training.shape[1]]);

	
	ConvNet = CNN( [patchWidth, patchHeight], 
	               outputDim, 
				   params_layers, 
				   epochs,
				   minibatch,
				   alpha,
				   momentum,
				   debug=debug);
	
	success = ConvNet.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(ConvNet.predict(images_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	# Check accuracy on test data
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_max_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_max_test);
	images_test = images_test / 255.0;
	images_test = np.reshape(images_test, [patchWidth, patchHeight, images_test.shape[1]]);
	
	correct_test = labels_test == np.argmax(ConvNet.predict(images_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	
	# See some samples
	for i in range(nVisSamples):
		pl.figure()
		sampleId = np.random.randint(images_test.shape[2])
		sampleImage = images_test[:,:,sampleId];
		pred = np.argmax(ConvNet.predict(np.transpose(np.array([sampleImage]),[1,2,0])), 0)
		pl.imshow(sampleImage, cmap='gray');
		pl.title("Prediction: " + str(pred))
		pl.axis('off')
		pl.show();