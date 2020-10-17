''' sparseae.py
	
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
from UFL.PCA import PCA

ACTIVATION_FUNCTION_SIGMOID = 0;
ACTIVATION_FUNCTION_IDENTITY = 1;
SPARSEAE_ACTIVATION_FUNCTIONS = [ACTIVATION_FUNCTION_SIGMOID, ACTIVATION_FUNCTION_IDENTITY]

class SparseAutoencoder:
	''' 
	Sparse Autoencoder
	
	Implemented as 3 layered neural network
	'''

	def __init__(self,
	             dimLayers,
				 lambda_w=0.0001,
				 beta=3,
				 sparsityParam=0.01,
				 actFunctions=[ACTIVATION_FUNCTION_SIGMOID, ACTIVATION_FUNCTION_SIGMOID],
				 debug=0):
		''' 
		Initialization function of the Sparse Autoencoder class
		
		Arguments
		dimLayers		: Dimensions of the layers, must be in the form [Input layer dim., hidden layer dim., output layer dim.] where input dim. = output dim.
		lambda_w		: weight decay parameter, default is 0.0001
		beta			: weight of sparsity penalty term, default is 3
		sparsityParam	: weight of the sparsity in the cost function, default is 0.01
		actFunctions	: activation function, default is [ACTIVATION_FUNCTION_SIGMOID, ACTIVATION_FUNCTION_SIGMOID]
		debug			: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.dimLayers = dimLayers;
		self.lambda_w = lambda_w;
		self.beta = beta;
		self.sparsityParam = sparsityParam;
		self.actFunctions = actFunctions;
		
		assert self.dimLayers[0]>0, 'ERROR:SparseAutoencoder:init: Input size must be >0'
		assert self.dimLayers[1]>0, 'ERROR:SparseAutoencoder:init: Feature size must be >0'
		assert self.dimLayers[0]==self.dimLayers[-1], 'ERROR:SparseAutoencoder:init: Input and output size should be the same'
		
		self.nLayers = len(self.dimLayers);
		
		assert self.nLayers==3, 'ERROR:SparseAutoencoder:init: Autoencoder should be 3 layered network'
		assert len(self.actFunctions)==(self.nLayers-1), 'ERROR:SparseAutoencoder:init: Please give activation functions for hidden and output layers'
		for f in self.actFunctions:
			assert f in SPARSEAE_ACTIVATION_FUNCTIONS, 'ERROR:SparseAutoencoder:init: Activation function not recognized'
		
		weights = [];
		biases = [];
		self.weightPrototypes = [];
		self.biasPrototypes = [];
		self.sizeParams = 0;
		# Choose weights uniformly from the interval [-r, r]
		r  = np.sqrt(6) / np.sqrt(self.dimLayers[0]+self.dimLayers[1]+1);
		for i in range(self.nLayers-1):
			weights.append( np.random.rand(self.dimLayers[i+1], self.dimLayers[i]) * 2 * r - r );
			biases.append( np.zeros((self.dimLayers[i+1], 1)) );
			# Set network topology
			self.weightPrototypes.append((self.dimLayers[i+1], self.dimLayers[i]));
			self.biasPrototypes.append((self.dimLayers[i+1], 1));
			# Total number of parameters
			self.sizeParams = self.sizeParams + (self.dimLayers[i+1] * self.dimLayers[i]) + (self.dimLayers[i+1]);
		
		self.params = self.rollParameters(weights, biases);
		
		if debug:
			print 'DEBUG:SparseAutoencoder:init: initialized for inputSize: ', self.dimLayers[0];
			print 'DEBUG:SparseAutoencoder:init: initialized for featureSize: ', self.dimLayers[1];
			print 'DEBUG:SparseAutoencoder:init: initialized for lambda_w: ', self.lambda_w;
			print 'DEBUG:SparseAutoencoder:init: initialized for beta: ', self.beta;
			print 'DEBUG:SparseAutoencoder:init: initialized for sparsityParam: ', self.sparsityParam;
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
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:SparseAutoencoder:rollParameters: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:SparseAutoencoder:rollParameters: bias dimension does not match the network topology';
		
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
		assert len(params)==self.sizeParams, 'ERROR:SparseAutoencoder:unrollParameters: Parameter size mismatch'
		
		weights = [];
		biases = [];
		read_start = 0;
		read_end = 0;
		
		for i in range(self.nLayers - 1):
			# set the end index for read
			read_end = read_start + self.dimLayers[i+1]*self.dimLayers[i];
			# read the weights for the current layer
			w = params[read_start:read_end];
			# reshape the weights
			w = np.reshape(w, (self.dimLayers[i+1], self.dimLayers[i]))
			weights.append( w );
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
		
	def getWeights(self):
		''' 
		Returns the list of weight matrices of the network
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getWeights: The instance is not properly initialized'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		return weights

	def getBiases(self):
		''' 
		Returns the list of bias vectors of the network
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getBiases: The instance is not properly initialized'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		return biases

	def getParameters(self):
		''' 
		Returns the model parameters of the network
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getParameters: The instance is not properly initialized'
		
		return  self.params

	def setWeights(self, weights_new):
		''' 
		Updates the weights of the model parameters of the network
		
		Arguments
		weights_new	: New weights to set
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setWeights: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters(weights_new, self.weightPrototypes), 'ERROR:SparseAutoencoder:setWeights: weight dimension does not match the network topology' ;

		[weights, biases] = self.unrollParameters(self.params);
		weights = weights_new;
		self.params = self.rollParameters(weights, biases);
		
	def setBiases(self, biases_new):
		''' 
		Updates the biases of the model parameters of the network
		
		Arguments
		biases_new	: New biases to set
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setBiases: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters(biases_new, self.biasPrototypes), 'ERROR:SparseAutoencoder:setBiases: bias dimension does not match the network topology' ;
		
		[weights, biases] = self.unrollParameters(self.params);
		biases = biases_new;
		self.params = self.rollParameters(weights, biases);
		
	def setParameters(self, weights, biases):
		''' 
		Updates the internal Sparse Autoencoder parameters with the given ones
		
		Arguments
		weights	: list of weights to set for the the network
		biases	: list of biases to set for the network
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setParams: The instance is not properly initialized'

		self.params = self.rollParameters(weights, biases);

	def getParameterSize(self):
		''' 
		Returns the size of the model parameters i.e. (weights and biases) of the network
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getParameterSize: The instance is not properly initialized'
		
		return self.sizeParams
		
	def doForwardPropagation(self, X, weights, biases):
		''' 
		Computes the forward propagation of the input in the network:
		
		Z{l+1} = W{l}*H{l} + B{l}
		H{l+1} = f(Z{l+1})
		
		where
		{l} and {l+1} denote layers,
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
		assert self.isInitialized, 'ERROR:SparseAutoencoder:doForwardPropagation: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters(weights, self.weightPrototypes), 'ERROR:SparseAutoencoder:doForwardPropagation: weight dimension does not match the network topology' ;
		assert AuxFunctions.checkNetworkParameters(biases, self.biasPrototypes), 'ERROR:SparseAutoencoder:doForwardPropagation: bias dimension does not match the network topology';
		
		outputs = [];
		activities = [];
		for layer in range(self.nLayers-1):
			
			if layer==0:
				x = X;
			else:
				x = activities[layer-1];
			
			z = np.dot(weights[layer], x) + np.repeat(biases[layer], x.shape[1], 1);
			
			if self.actFunctions[layer]==ACTIVATION_FUNCTION_SIGMOID:
				h = AuxFunctions.sigmoid(z);
			elif self.actFunctions[layer]==ACTIVATION_FUNCTION_IDENTITY:
				h = 1.0 * z;
			else:
				# Should not be here
				print 'ERROR:SparseAutoencoder:doForwardPropagation: Activation function not recognized'
				sys.exit()
				
			outputs.append(z);
			activities.append(h);
		
		return [outputs, activities];
		
	def computeCost(self, theta, X):
		''' 
		Computes the value of the Sparse Autoencoder objective function for given 
		features (theta), data matrix (X) and corresponding labels (y):
		
		f = 1/2 * sum((X - H)^2) + beta * sum_j(KL(rho||rho_hat_j))
		
		where H is the activity matrix with columns corresponding to the samples and 
		rows corresponding to the output layer units,
		rho is the target average activity value,
		rho_hat_j is the average activity of unit j in the hidden layer:
		
		rho_hat_j = 1/nSamples * sum_i(h_j(x_i)),		i=1...nSamples
		
		h_j(x_i) is the activity of hidden unit j given input x_i,
		KL(.) is Kullback-Leibler divergence function:
		
		KL(rho||rho_hat) = rho*log(rho/rho_hat) + (1-rho)*log((1-rho)/(1-rho_hat)).
		
		KL function penalizes any average values that diverges from rho	hence, sparsity 
		in the hidden units is encouraged.
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:computeCost: The instance is not properly initialized'
		
		f = 0;
		
		nSamples = X.shape[1];
		
		[weights, biases] = self.unrollParameters(theta);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);
		
		cost_fidelty = (1.0/nSamples) * np.sum( 0.5 * (X - activities[-1])**2 );
		
		W_sum = 0;
		for l in range(len(weights)):
			W_sum = W_sum + np.sum((weights[l])**2);
			
		cost_regularization = (self.lambda_w/2) * W_sum;
		
		a2_mean = np.mean(activities[0], 1);
		cost_sparsity = self.beta * np.sum(self.sparsityParam * np.log(self.sparsityParam/a2_mean) + (1 - self.sparsityParam) * np.log( (1 - self.sparsityParam)/(1 - a2_mean) ));
		
		f = cost_fidelty + cost_regularization + cost_sparsity;
		
		return f
		
	def computeGradient(self, theta, X):
		''' 
		Computes gradients of the Sparse Autoencoder objective function for given parameters,
		data and corresponding labels using error back propagation:
		
		E_{3} = (H_{3} - X)
		E_{2} = (W_{2}'*E_{3}  + beta*(-rho/rho_hat + (1-rho)/(1-rho_hat)) ) * df(Z_{2})/dz
		
		where rho is the target average activity value,
		rho_hat is the average activity vector of the units in the hidden layer,
		df(z)/dz is the derivatives of acitvation function f(.) at points Z which is
		
		df(Z)/dz = f(Z)*(1-f(Z)) 	for sigmoid activation function and
		df(Z)/dz = Z 				for identity activation function. 
		
		Gradients are computed via:
		
		dJ(W,b;X,y)/dW_{l} = E_{l+1} * H_{l}'
		dJ(W,b;X,y)/db_{l} = sum(E_{l+1})
		
		where sum(.) is taken columnwise i.e. over samples
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples].
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:computeGradient: The instance is not properly initialized'
		
		nSamples = X.shape[1];
		
		[weights, biases] = self.unrollParameters(theta);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);
		
		# Output layer
		if self.actFunctions[-1]==ACTIVATION_FUNCTION_SIGMOID:
			delta3 = 1.0/nSamples * (activities[-1] - X) * ( activities[-1] * (1 - activities[-1]) );
		elif self.actFunctions[-1]==ACTIVATION_FUNCTION_IDENTITY:
			delta3 = 1.0/nSamples * (activities[-1] - X);
		else:
			# Should not be here
			print 'ERROR:SparseAutoencoder:doForwardPropagation: Activation function not recognized'
			sys.exit()
	
		# Hidden layer
		if self.actFunctions[-2]==ACTIVATION_FUNCTION_SIGMOID:
			aux1 = np.dot(np.transpose(weights[1]), delta3);
			a2_mean = np.mean(activities[0], 1);
			aux2 = self.beta/(nSamples*1.0) * ( (1-self.sparsityParam)/(1-a2_mean) - (self.sparsityParam/a2_mean) );
			aux3 = np.repeat(np.reshape(aux2, [len(aux2), 1]), nSamples, 1 );
			delta2 = (aux1 + aux3) * ( activities[-2] * (1 - activities[-2]) );
		else:
			# Should not be here
			print 'ERROR:SparseAutoencoder:doForwardPropagation: Activation function not recognized'
			sys.exit()
		
		deltas = [delta2, delta3];
		gradients_W = [];
		gradients_b = [];
		for layer in range(len(weights)):
			if layer==0:
				x_in = X;
			else:
				x_in = activities[layer-1];
			
			gradients_W.append( np.dot(deltas[layer], np.transpose(x_in)) + (self.lambda_w * weights[layer]) );
			gradients_b.append( np.sum(deltas[layer], 1) );
		
		return self.rollParameters(gradients_W, gradients_b);
		
	def testGradient(self, X):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [input dim., number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SparseAutoencoder:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:SparseAutoencoder:testGradient: Testing gradient computation...'
		
		result = 0;
		
		grad = self.computeGradient(self.params, X);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.params, args=((X,)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:SparseAutoencoder:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseAutoencoder:testGradient:Gradient check PASSED!'
				print
				
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:SparseAutoencoder:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseAutoencoder:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
	
	def optimizeParameters(self, X):
		'''
		Optimizes the parameters of the Sparse Autoencoder model
		
		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SparseAutoencoder:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print "DEBUG:SparseAutoencoder:optimizeParameters: Optimizing parameters..."
		
		# Set optimization options
		method = 'L-BFGS-B'
		options = {};
		options['maxiter'] = 400;

		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.params, args=(X,), method=method, options=options)
		
		# Set the new values
		self.params = result.x;
		
		if self.debug: print 'DEBUG:SparseAutoencoder:optimizeParameters: Optimization result: ', result.message
		
		return result.success;

	def predict(self, X):
		'''
		Applies the Sparse Autoencoder model to the given data
		
		Arguments
		X		: data in the form [input dim., number of samples]
		W		: weights of the first layer (optional)
		b 		: biases of the first layer (optional)
		
		Returns
		output of the network in the form [output dim., number of samples]
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:predict: The instance is not properly initialized'
		assert X.shape[0]==self.dimLayers[0], 'ERROR:SparseAutoencoder:predict: Dimensions of given data do not match with the number of parameters'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);

		return activities[0];

	def doForwardPropagation_asLayer(self, X, theta):
		''' 
		Wrapper function for doForwardPropagation for cases where Spaerse Autoencoder is
		used as a layer of a deep network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		theta		: model parameters for the first layer, must be packed as [weights+biases]
		
		Returns
		activation	: activation if the first layer
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:doForwardPropagation_asLayer: The instance is not properly initialized'
		assert np.size(theta)==self.weightPrototypes[0][0]*self.weightPrototypes[0][1]+self.biasPrototypes[0][0]*self.biasPrototypes[0][1], 'ERROR:SparseAutoencoder:doForwardPropagation_asLayer: Dimensions of the given model parameter do not match the internal structure'

		theta_local = self.params;
		theta_local[0:np.size(theta)] = theta;
		[weights, biases] = self.unrollParameters(theta_local);
		
		[outputs, activities] = self.doForwardPropagation(X, weights, biases);
		
		return activities[0];

	def doBackPropagateError_asLayer(self, error, theta, layer_in, layer_out):
		'''
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:doBackPropagateError_asLayer: The instance is not properly initialized'
		assert np.size(theta)==self.weightPrototypes[0][0]*self.weightPrototypes[0][1]+self.biasPrototypes[0][0]*self.biasPrototypes[0][1], 'ERROR:SparseAutoencoder:doBackPropagateError_asLayer: Dimensions of the given model parameter do not match the internal structure'
		
		weights = np.reshape(theta[0:self.weightPrototypes[0][0]*self.weightPrototypes[0][1]], self.weightPrototypes[0]);
		
		delta = error * (layer_out * (1 - layer_out))
		
		gradients_W = np.dot(delta, np.transpose(layer_in));
		gradients_b = np.sum(delta, 1);
		
		grad = np.hstack((gradients_W.flatten(), gradients_b.flatten()));
		
		error_prop = np.dot(np.transpose(weights), delta);
		
		return grad, error_prop;
		
	def getParameterSize_asLayer(self):
		''' 
		Wrapper function for getParameterSize for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getParameterSize: The instance is not properly initialized'
		
		return self.weightPrototypes[0][0] * self.weightPrototypes[0][1] + self.biasPrototypes[0][0] * self.biasPrototypes[0][1]
		
	def getWeights_asLayer(self):
		''' 
		Wrapper function for getWeights for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getWeights_asLayer: The instance is not properly initialized'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		return weights[0]

	def getBiases_asLayer(self, layer=0):
		''' 
		Wrapper function for getBiases for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getBiases_asLayer: The instance is not properly initialized'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		return biases[0]
	
	def getParameters_asLayer(self):
		''' 
		Wrapper function for getParameters for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:getParameters_asLayer: The instance is not properly initialized'
		
		[weights, biases] = self.unrollParameters(self.params);
		
		return  np.hstack((weights[0].flatten(), biases[0].flatten()))
	
	def setWeights_asLayer(self, W):
		''' 
		Wrapper function for setWeights for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setWeights_asLayer: The instance is not properly initialized'

		[weights, biases] = self.unrollParameters(self.params);
		weights[0] = W;
		
		self.params = self.rollParameters(weights, biases);
		
	def setBiases_asLayer(self, b):
		''' 
		Wrapper function for setBiases for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setBiases_asLayer: The instance is not properly initialized'

		[weights, biases] = self.unrollParameters(self.params);
		biases[0] = b;
		self.params = self.rollParameters(weights, biases);

	def setParameters_asLayer(self, theta):
		''' 
		Wrapper function for setParameters for cases where Spaerse Autoencoder is
		used as a layer of a deep network. Discards the second layer.
		'''
		assert self.isInitialized, 'ERROR:SparseAutoencoder:setParameters_asLayer: The instance is not properly initialized'
		assert np.size(theta)==np.size(self.params[0:(self.weightPrototypes[0][0]*self.weightPrototypes[0][1]+self.biasPrototypes[0][0]*self.biasPrototypes[0][1])])
		
		self.params[0:(self.weightPrototypes[0][0]*self.weightPrototypes[0][1]+self.biasPrototypes[0][0]*self.biasPrototypes[0][1])] = theta;
				
if __name__ == '__main__':
	
	# Examples:
	# 1) Sparse Autoencoder example on natural images
	# 2) Linear decoder example on color images
	example = 2; 
	
	# Test gradient computation?
	doTest = True;
	
	if (doTest):
		debug 			= 2;
		numPatches 		= 10;
		patchWidth 		= 2;
		patchHeight 	= 2;
		imChannels		= 1;
		inputDim 		= patchWidth * patchHeight * imChannels;
		numFeatures 	= 5;
		
		patches = np.random.rand(inputDim, numPatches);
		
		dimLayers = [inputDim, numFeatures, inputDim];
		
		SAE_Test = SparseAutoencoder(dimLayers=dimLayers, debug=debug);
		
		print 'Checking gradient...'
		
		SAE_Test.testGradient(patches);
		
	if(example==1):
		
		print "-------------------"
		print "Autoencoder Example"
		print "-------------------"
		
		if 1:
		  filename_data = '/home/cem/develop/UFL/data/stlSampledPatches.mat';
		else:
		  filename_data = 'C://develop//python//UFL//data//stlSampledPatches.mat';
		
		debug 			= 1;
		numPatches 		= 10000;
		patchWidth 		= 8;
		patchHeight 	= 8;
		imChannels		= 1;
		inputDim 		= patchWidth * patchHeight * imChannels;
		numFeatures 	= 25;
		lambda_w 		= 0.0001;     # weight decay parameter       
		beta 			= 3;          # weight of sparsity penalty term       
		sparsityParam 	= 0.01;
		actFunctions	= [ACTIVATION_FUNCTION_SIGMOID, ACTIVATION_FUNCTION_SIGMOID]
		
		# Read data from file
		data = scipy.io.loadmat(filename_data);
		
		patches = DataInputOutput.samplePatches(data['IMAGES'], patchWidth, patchHeight, numPatches);
		
		# Normalize data
		patches = DataNormalization.normZeroToOne(patches);
		
		if debug>1:
			Visualization.displayNetwork(patches[:, 0:100]);
		
		if debug:
			print 'Number of samples: ', patches.shape[1];
			
		dimLayers = [inputDim, numFeatures, inputDim];
		
		SAE = SparseAutoencoder(dimLayers=dimLayers, lambda_w=lambda_w, beta=beta, sparsityParam=sparsityParam, actFunctions=actFunctions, debug=debug);
		
		success = SAE.optimizeParameters(patches);
		
		# Visualize the learned bases
		weightMatrix = SAE.getWeights();
		Visualization.displayNetwork(np.transpose(weightMatrix));
		
	elif(example==2):
		
		print "----------------------"
		print "Linear Decoder Example"
		print "----------------------"
		
		if 1:
		  filename_data = '/home/cem/develop/UFL/data/stlSampledPatches.mat';
		else:
		  filename_data = 'C://develop//python//UFL//data//stlSampledPatches.mat';
		
		debug 			= 1;
		numPatches 		= 10000;
		patchWidth 		= 8;
		patchHeight 	= 8;
		imChannels		= 3;
		inputDim 		= patchWidth * patchHeight * imChannels;
		numFeatures 	= 400;
		lambda_w 		= 3e-3;		# weight decay parameter       
		beta 			= 5;		# weight of sparsity penalty term       
		sparsityParam 	= 0.035;
		actFunctions	= [ACTIVATION_FUNCTION_SIGMOID, ACTIVATION_FUNCTION_IDENTITY]
		epsilon 		= 0.1		# Epsilon for ZCA whitening
		
		# Read data from file
		data = scipy.io.loadmat(filename_data);
		patches = data['patches'][:, 0:numPatches];
		
		instance_pca = PCA.PCA(inputDim, epsilon=epsilon, debug=debug);
		ZCAWhite = instance_pca.computeZCAWhiteningMatrix(patches);
		patches_ZCAwhite = instance_pca.doZCAWhitening(patches);
		
		if debug>1:
			Visualization.displayColorNetwork(patches[:, 0:100]);
		
		if debug:
			print 'Number of samples: ', patches.shape[1];
			
		dimLayers = [inputDim, numFeatures, inputDim];
		
		SAE = SparseAutoencoder(dimLayers=dimLayers, lambda_w=lambda_w, beta=beta, sparsityParam=sparsityParam, actFunctions=actFunctions, debug=debug);
		
		success = SAE.optimizeParameters(patches_ZCAwhite);
		
		# Visualize the learned bases
		weightMatrix = SAE.getWeights();
		Visualization.displayColorNetwork(np.transpose(np.dot(weightMatrix, ZCAWhite)));