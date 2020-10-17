''' StackedAE.py
	
	Implementation of Deep Neural Network
	
	Author: Cem Karaoguz
	Date: 13.03.2015
	Version: 1.0
	
	TODO: implement Layers.py
	TODO: implement SoftICA
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize
import scipy.io
import scipy.linalg

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization
from UFL.SparseAutoencoder import SparseAutoencoder
from UFL.SoftICA import SoftICA
from UFL.Softmax import Softmax

DEEPNETWORK_LAYER_TYPES 		= ['sparseae', 'softica', 'softmax']
DEEPNETWORK_INDEX_LAYER_INPUT	= 0;
DEEPNETWORK_INDEX_LAYER_OUTPUT	= -1;

class DeepNetwork:
	''' 
	Deep Neural Network
	
	Neural Network consisting of multiple layers.
	A specific example of a deep network is stacked autoencoders
	that is in the form:
	
	Input -> AE -> AE -> ... -> Softmax -> Output
	
	Other variants are also possible:
	
	Input -> SoftICA -> Softmax -> Output
	
	Usually layers are trained separately and combined later on.
	Thereafter it is possible to perform fine tuning of the whole
	network via back propagation.
	'''

	def __init__(self,
	             inputDim,
				 layerParams,
				 lambd=1e-4,
				 doFineTuning=True,
				 debug=0):
		''' 
		Initialization function of the Deep Network class
		
		Arguments		
		inputDim			: Dimension of the input layer
		layerParams			: Parameters of the layers
		lambd				: weight decay parameter, default is 0.003
		doFineTuning		: Do fine tuning optimization step?
		debug				: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.lambd = lambd;
		self.doFineTuning = doFineTuning;
		self.layerParams = layerParams;
		
		assert self.inputDim>0, 'ERROR:DeepNetwork:init: Input layer dimension must be >0'
		
		# Check necessary keys inside the hidden layer parameters
		for p in layerParams:
			assert ('id' in p.keys()), 'ERROR:DeepNetwork:init: ID field is mandatory for hidden layer parameter'
			assert ('featureDim' in p.keys()), 'ERROR:DeepNetwork:init: featureDim field is mandatory for hidden layer parameter'
			if not (p['id'] in DEEPNETWORK_LAYER_TYPES):
				print 'ERROR:DeepNetwork:init:', p['id'], 'is not allowed. Valid ids are:\n', DEEPNETWORK_LAYER_TYPES
				sys.exit();
		
		layerDims = []
		layerDims.append(self.inputDim);
		for i in range(len(self.layerParams)):
			layerDims.append(layerParams[i]['featureDim'])
		
		# Initialize layers and set dimensions
		self.layers = []
		self.modelParameterPrototype = []
		for layer in range(len(self.layerParams)):
			layerparams = self.layerParams[layer];
			if layerparams['id']=='sparseae':
				dimLayers = [layerDims[layer], layerDims[layer+1], layerDims[layer]];
				sae = SparseAutoencoder.SparseAutoencoder(dimLayers=dimLayers, 
														  lambda_w=layerparams['lambd'], 
														  beta=layerparams['beta'], 
														  sparsityParam=layerparams['sparsityParam'], 
														  actFunctions=layerparams['actFunctions'], 
														  debug=layerparams['debug']);
				self.layers.append(sae);
				self.modelParameterPrototype.append(sae.getParameterSize_asLayer());
				
			elif layerparams['id']=='softica':
				dimLayers = [layerDims[layer], layerDims[layer+1]];
				sica = SoftICA.SoftICA(dimLayers, 
				                       lambd=layerparams['lambd'], 
									   epsilon=layerparams['epsilon'], 
									   debug=layerparams['debug']);
				self.layers.append(sica)
				self.modelParameterPrototype.append(sica.getParameterSize_asLayer());
			
			elif layerparams['id']=='softmax':
				softmaxmodel = Softmax.Softmax(layerparams['featureDim'], 
											   layerparams['outputDim'], 
											   debug=layerparams['debug']);
				self.layers.append(softmaxmodel)
				self.modelParameterPrototype.append(softmaxmodel.getParameterSize_asLayer());
		
			else:
				print 'ERROR:DeepNetwork:init: identity ', layerparams['id'], ' is not recognized for layer:', layer;
				sys.exit();
		
		# Check if all required functions exist
		required_functions = ['getParameters_asLayer',
		                      'setParameters_asLayer',
							  'doForwardPropagation_asLayer',
							  'doBackPropagateError_asLayer',
							  'optimizeParameters',
							  'computeCost',
							  'predict' ];
		
		for i in range(len(self.layers)):
			assert callable(getattr(self.layers[i], required_functions[i], None)), 'ERROR:DeepNetwork:init: All layers require function: ' + required_functions[i]
			
		if debug:
			print 'DEBUG:DeepNetwork:init: initialized for lambd: ', self.lambd;
			print 'DEBUG:DeepNetwork:init: initialized for doFineTuning?: ', self.doFineTuning;
			print
		
		self.isInitialized = True;
		
	def getNetworkParameters(self):
		''' 
		Returns the rolled model parameters of each layer of the network in a list
		'''
		theta = [];
		for layer in self.layers:
			layerparams = layer.getParameters_asLayer();
			theta.append(layerparams);
		
		return theta
	
	def setNetworkParameters(self, theta):
		''' 
		Updates the internal Deep Network model parameters of each layer with the given ones
		
		Arguments
		theta	: rolled model parameters to set for the first layer of the network
		'''
		i = 0;
		for layer in self.layers:
			layer.setParameters_asLayer(theta[i]);
			i = i + 1;

	def unstackParameters(self, theta_list):
		''' 
		Converts the model parameters from stacked form into vector form
		
		Arguments
		theta_list	: list of model parameters of each layer 
		
		Returns
		theta		: vector of combined model parameters of the network
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:unstackParameters: The instance is not properly initialized'
		assert AuxFunctions.checkNetworkParameters(theta_list, self.modelParameterPrototype), 'ERROR:DeepNetwork:unstackParameters: model parameter dimension does not match the network topology'
		
		theta = np.array([]);
		for i in range(len(theta_list)):
			theta = np.hstack((theta, theta_list[i].flatten()))
		
		return theta
		
	def stackParameters(self, theta):
		''' 
		Converts the model parameters from vector form into stacked form
		
		Arguments
		theta		: vector of combined model parameters of the network
		
		Returns
		theta_list	: list of model parameters of each layer 
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:stackParameters: The instance is not properly initialized'
		
		theta_list = [];
		i_start = 0;
		for i in range(len(self.modelParameterPrototype)):
			i_stop = i_start + self.modelParameterPrototype[i];
			theta_list.append(theta[i_start:i_stop])
			i_start = i_stop;
		
		assert AuxFunctions.checkNetworkParameters(theta_list, self.modelParameterPrototype), 'ERROR:DeepNetwork:stackParameters: model parameter dimension does not match the network topology'
		
		return theta_list
		
	def doForwardPropagation(self, X, theta_list):
		''' 
		Computes the forward propagation of the input in the network.
		
		Arguments
		X			: data matrix in the form [input dim., number of samples]
		theta_list	: list of model parameters of each layer
		
		Returns
		activations	: list of activation matrices (h) of each layer (output of neuron after activation function)
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:doForwardPropagation: The instance is not properly initialized'
		
		activations = [];
		indata = X;
		i = 0;
		for layer in self.layers:
			layer_act = layer.doForwardPropagation_asLayer(indata, theta_list[i]);
			
			activations.append(layer_act)
			indata = layer_act;
			i = i + 1;
		
		return activations;
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the Deep Network cost function for given 
		model parameters (theta), data matrix (X) and corresponding labels (y).
		This is merely the cost of the output layer.
		
		Arguments
		theta	: network model parameters in rolled form
		X		: data matrix in the form [input dim, number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:computeCost: The instance is not properly initialized'
		
		f = 0;
		
		nSamples = X.shape[1];
		
		theta_list = self.stackParameters(theta);
		
		activations = self.doForwardPropagation(X, theta_list);
		
		# Get the cost of the last layer, for that, set input, output and model parameters properly
		f = self.layers[DEEPNETWORK_INDEX_LAYER_OUTPUT].computeCost( theta_list[DEEPNETWORK_INDEX_LAYER_OUTPUT],
		                                                             activations[DEEPNETWORK_INDEX_LAYER_OUTPUT-1],
				    												 y );
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the Deep Network objective function wrt parameters
		(theta) for a given data matrix (X) and corresponding labels (y) with error
		back propagation. The following steps are followed:
		
		1) Error in the output of the network is computed
		2) Error is back-propagated through each layer and gradients are computed
		using each layer's own private method.
		
		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (number of parameters*number of classes,)
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:computeGradient: The instance is not properly initialized'
		
		gradients = [];
		nSamples = X.shape[1];
		
		theta_list = self.stackParameters(theta);
		
		activations = self.doForwardPropagation(X, theta_list);
		
		outputDim = np.shape(activations[DEEPNETWORK_INDEX_LAYER_OUTPUT])[0];
		
		# Compute output error
		aux1 = np.repeat(np.reshape(range(outputDim), [1, outputDim]), nSamples, 0)
		aux2 = np.repeat(np.reshape(y, [nSamples, 1]), outputDim, 1);
		groundTruth = np.transpose((aux1==aux2).astype(int));
		error = (-1.0/nSamples)*(groundTruth - activations[DEEPNETWORK_INDEX_LAYER_OUTPUT]);
		
		# Back propagate output error in layers
		for i in range(len(self.layers)-1, -1, -1):
		
			# Layer input
			if i==0:
				layer_in = X;
			else:
				layer_in = activations[i-1];
			
			# Layer output
			layer_out = activations[i];
			
			[grads, error] = self.layers[i].doBackPropagateError_asLayer(error, theta_list[i], layer_in, layer_out);
			
			gradients.append(grads)
		
		gradients = list(reversed(gradients));
		
		return self.unstackParameters(gradients);
		
	def testGradient(self, X, y):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [input dim., number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:testGradient: The instance is not properly initialized'
		
		if self.debug: print 'DEBUG:DeepNetwork:testGradient: Testing gradient computation...'
		
		result = 0;
		
		theta_list = self.getNetworkParameters();
		theta = self.unstackParameters(theta_list);
		
		grad = self.computeGradient(theta, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=theta, args=((X, y)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:DeepNetwork:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:DeepNetwork:testGradient:Gradient check PASSED!'
				print
			
			result = 0;
			
		else:
			if self.debug:
				print 'DEBUG:DeepNetwork:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:DeepNetwork:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
			
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the Deep Network model
		
		Optimization is done in two steps:
		1) Optimization of each layers individually
		2) Fine tuning of the whole network using error back-propagation.
		
		Arguments
		X		: data in the form [input dim., number of samples]
		y		: labels in the form [1, number of samples].
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:DeepNetwork:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print "DEBUG:DeepNetwork:optimizeParameters: Optimizing parameters..."
		
		result = 0;
		
		indata = X;
		for i in range(len(self.layers)):
			
			if debug: print "DEBUG:DeepNetwork:optimizeParameters: Training hidden layer ", i+1, ': ', self.layerParams[i]['id']
			
			# Optimize the cost function
			if self.layerParams[i]['id']=='sparseae':
				result = self.layers[i].optimizeParameters(indata);
			if self.layerParams[i]['id']=='softica':
				result = self.layers[i].optimizeParameters(indata);
			if self.layerParams[i]['id']=='softmax':
				result = self.layers[i].optimizeParameters(indata, y);
			
			# Set the output of the current autoencoder as the input of the next layer
			indata = self.layers[i].predict(indata);
		
		# Get new network parameters
		theta_list = self.getNetworkParameters();
		theta = self.unstackParameters(theta_list);
		
		if self.doFineTuning:
			if debug: print "DEBUG:DeepNetwork:optimizeParameters: Fine tuning the deep network..."
			
			# Set optimization options
			method = 'L-BFGS-B'
			options = {};
			options['maxiter'] = 100;
			
			if self.debug:
				options['disp'] = True;
			else:
				options['disp'] = False;
			
			# Optimize the cost function
			result_finetune = minimize(fun=self.computeCost, jac=self.computeGradient, x0=theta, args=(X,y), method=method, options=options)
			
			result = result_finetune.success;
			
			if self.debug:
				print 'DEBUG:DeepNetwork:optimizeParameters:',  result_finetune.message
			
			# Set the new values
			self.setNetworkParameters(self.stackParameters(result_finetune.x));
			
		return result;

	def predict(self, X):
		'''
		Applies the Deep Network model to the given data
		
		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		output of the network in the form [output dim., number of samples]
		'''
		assert self.isInitialized, 'ERROR:DeepNetwork:predict: The instance is not properly initialized'
		
		theta_list = self.getNetworkParameters();
		
		activities = self.doForwardPropagation(X, theta_list);

		return activities[DEEPNETWORK_INDEX_LAYER_OUTPUT];
	
	
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
		
		# Parameters for softmax
		params_softmax					= {}
		params_softmax['id']			= 'softmax'
		params_softmax['outputDim']	 	= 2;
		params_softmax['featureDim']	= 5;
		params_softmax['debug']			= 1;
		
		layerParams = [params_ae, params_ae, params_softmax];
		
		#testdata = np.random.rand(inputDim, numPatches);
		#testlabel = np.round(np.random.rand(numPatches));
		testlabel = np.array([1,1,1,1,1,2,2,2,2,2])-1;
		testdata = np.transpose(np.reshape(range(inputDim*numPatches), [numPatches,inputDim]));
		
		StackedAE_test = DeepNetwork( inputDim, layerParams, debug=debug);
		
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

	# Parameters for softmax
	params_softmax					= {}
	params_softmax['id']			= 'softmax'
	params_softmax['outputDim']	 	= outputDim;
	params_softmax['featureDim']	= 200;
	params_softmax['debug']			= 1;
	
	layerParams = [params_ae, params_ae, params_softmax];
	
	# Read data from file
	labels_training = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_training, nSamples_max_train);	
	images_training = DataInputOutput.loadMNISTImages(mnist_img_filename_training, nSamples_max_train);
	labels_test = DataInputOutput.loadMNISTLabels(mnist_lbl_filename_test, nSamples_max_test);	
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_max_test);

	# Normalize data 
	images_training = images_training / 255.0;
	images_test = images_test / 255.0;
	
	DNN = DeepNetwork( inputDim,
					   layerParams,
					   lambd=lambd,
					   doFineTuning=doFineTuning,
					   debug=debug);
	
	success = DNN.optimizeParameters(images_training, labels_training);
	
	# Print out accuracy
	correct_training = labels_training == np.argmax(DNN.predict(images_training),0)
	accuracy_training = np.sum(correct_training.astype(int)) * 100 / len(labels_training);
	print 'Training accuracy: ', accuracy_training, '%'
	
	correct_test = labels_test == np.argmax(DNN.predict(images_test),0)
	accuracy_test = np.sum(correct_test.astype(int)) * 100 / len(labels_test);
	print 'Test accuracy: ', accuracy_test, '%'
	
