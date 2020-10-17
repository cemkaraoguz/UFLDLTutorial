''' ICA.py
	
	Implementation of Independent Component Analysis
	
	Author: Cem Karaoguz
	Date: 13.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
import scipy.io
import scipy.linalg

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization
from UFL.PCA import PCA

ICA_COST_FUNCTION_ICA = 0;
ICA_COST_FUNCTION_RICA = 1;
ICA_COST_FUNCTIONS = [ICA_COST_FUNCTION_ICA, ICA_COST_FUNCTION_RICA]

class ICA:
	''' 
	Independent Component Analysis
	
	Implementation is similar to sparse Autoencoder network where the input 
	data is projected on a lower dimensional space of bases which are then 
	projected back to the output layer with the dimension same as the original data.
	
	Output of the network is the same as the input (i.e. reconstructive)
	Cost function is the L1 norm between the input and output layers with an
	additional constraint of the orthagonality of basis vectors.
	'''

	def __init__(self,
	             sizeLayers,
				 alpha=0.5,
				 epsilon=1e-6,
				 maxIterations=3000,
				 costFunction=ICA_COST_FUNCTION_RICA,
				 debug=0):
		''' 
		Initialization function of the ICA class
		
		Arguments
		sizeLayers		: Size of the layers, must be in the form [Input layer size, hidden layer size, output layer size]
						  where output layer size = input layer size
		alpha			: Backtracking line search learning rate, default is 0.5
		epsilon			: L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon), default is 1e-6
		maxIterations	: Backtracking line search maximum number of iterations, default is 30000
		costFunction	: Cost function, [ICA_COST_FUNCTION_ICA, ICA_COST_FUNCTION_RICA*]
		debug			: Debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = sizeLayers[0];
		self.featureDim = sizeLayers[1];
		self.costFunction = costFunction;
		self.alpha = alpha;
		self.epsilon = epsilon;
		self.maxIterations = maxIterations;
		
		assert self.inputDim>0, 'ERROR:ICA:init: Input size must be >0'
		assert self.featureDim>0, 'ERROR:ICA:init: Feature size must be >0'
		assert self.costFunction in ICA_COST_FUNCTIONS, 'ERROR:ICA:init: Cost function invalid'
		
		self.params = np.random.rand(self.featureDim*self.inputDim);
		
		self.weightMatrixPrototype = [self.featureDim, self.inputDim];
		
		if debug:
			print ('DEBUG:ICA:init: initialized for inputDim: ', self.inputDim);
			print ('DEBUG:ICA:init: initialized for featureDim: ', self.featureDim);
			print ('DEBUG:ICA:init: initialized for costFunction: ', self.costFunction);
			print ('DEBUG:ICA:init: initialized for alpha: ', self.alpha);
			print ('DEBUG:ICA:init: initialized for epsilon: ', self.epsilon);
			print ('DEBUG:ICA:init: initialized for maxIterations: ', self.maxIterations);
			print()
		
		self.isInitialized = True;
		
	def rollParameters(self, theta):
		''' 
		Converts a given parameter matrix into a vector
		
		Arguments
		theta	: parameter matrix
		
		Returns
		theta	: parameter vector
		'''
		assert self.isInitialized, 'ERROR:ICA:rollParameters: The instance is not properly initialized'
		
		return theta.flatten();
		
	def unrollParameters(self, theta):
		''' 
		Converts the vectorized parameters into matrix
		
		Arguments
		theta	: parameter vector
		
		Returns
		theta	: parameter matrix
		'''
		assert self.isInitialized, 'ERROR:ICA:unrollParameters: The instance is not properly initialized'
		assert len(theta)==self.featureDim*self.inputDim, 'ERROR:ICA:unrollParameters: dimensions of given parameters do not match internal parameter structure'

		return np.reshape(theta, self.weightMatrixPrototype);
		
	def getWeights(self):
		''' 
		Returns the ICA weight matrix
		'''
		return self.unrollParameters(self.params)
		
	def computeCost(self, theta, X):
		''' 
		Computes the value of the ICA objective function for given features
		(theta) and data matrix (X). Two cost functions are implemented:
		
		ICA_COST_FUNCTION_ICA		f = || theta*X ||_1
		
		ICA_COST_FUNCTION_RICA		f = lambda * || theta*X ||_1 + 1/2(|| theta'*theta*X - X ||_2)^2
		
		where
		
		|| . ||_1 denotes L1 norm and || . ||_2 denotes L2 norm
		
		In addition to the sparsity cost in ICA, RICA employs a reconstruction cost that drives orthanormality
		of basis vectors without imposing a hard constraint (as done in ICA)
		
		Arguments
		theta	: function parameters in the form (feature dim * input dim, )
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		f		: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:ICA:computeCost: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:ICA:computeCost: Dimensions of given data do not match with the number of parameters'
		
		f = 0;
		
		nSamples = X.shape[1];
		
		weightMatrix = self.unrollParameters(theta);
		
		if self.costFunction==ICA_COST_FUNCTION_ICA:
			
			aux1 = np.dot(weightMatrix, X);
			aux2 = np.sqrt((aux1**2) + self.epsilon);
			f = np.sum(aux2)/nSamples;
			
		elif self.costFunction==ICA_COST_FUNCTION_RICA:
			
			aux1 = np.sqrt(( np.dot(weightMatrix, X)**2) + self.epsilon);
			aux2 = np.dot(np.dot(np.transpose(weightMatrix), weightMatrix), X) - X;
			
			f = 1.0/nSamples * ( np.sum(aux1) + np.sum(aux2**2));
			
		else:
			'ERROR:ICA:computeCost: Cost function invalid!'
			sys.exit();
		
		return f
		
	def computeGradient(self, theta, X):
		''' 
		Computes gradients of the ICA objective function for given features
		(theta) and data matrix (X). Gradients for two cost functions are implemented:
		
		ICA_COST_FUNCTION_ICA		g = (theta*X)/sqrt(theta*X + epsilon)*X'
		
		ICA_COST_FUNCTION_RICA		g = (theta*X)/sqrt(theta*X + epsilon)*X' + 
		                                theta*(theta'*theta*X - X)*X' + 
										theta*X*(theta'*theta*X - X)'
		
		Arguments
		theta	: function parameters in the form [feature dim, input dim]
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (feature dim * input dim,)
		'''
		assert self.isInitialized, 'ERROR:ICA:computeGradient: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:ICA:computeCost: Dimensions of given data do not match with the number of parameters'

		nSamples = X.shape[1];
		
		weightMatrix = self.unrollParameters(theta);
		
		if self.costFunction==ICA_COST_FUNCTION_ICA:
			
			aux1 = np.dot(weightMatrix, X);
			aux2 = np.sqrt((aux1**2) + self.epsilon);
			aux3 = np.dot((aux1/aux2), np.transpose(X));
			grad = (aux3/nSamples);

		elif self.costFunction==ICA_COST_FUNCTION_RICA:
		
			aux1 = np.sqrt(( np.dot(weightMatrix, X)**2) + self.epsilon);
			aux2 = np.dot(np.dot(np.transpose(weightMatrix), weightMatrix), X) - X;
			
			grad_term1 = 1.0/nSamples * np.dot((np.dot(weightMatrix, X)/aux1), np.transpose(X));
			grad_term2 = 2.0/nSamples * ( np.dot(np.dot(weightMatrix, aux2), np.transpose(X)) + np.dot(np.dot(weightMatrix, X), np.transpose(aux2)) );
			
			grad = grad_term1 + grad_term2;
			
		else:
			'ERROR:ICA:computeCost: Cost function invalid!'
			sys.exit();
			
		return grad.flatten();
	
	def testGradient(self, X):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:ICA:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:ICA:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print ('DEBUG:ICA:testGradient: Testing gradient computation...')
		
		result = 0;
		
		grad = self.computeGradient(self.params, X);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.params, args=((X,)) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print ('DEBUG:ICA:testGradient:Gradient error: ', errorGrad)
				print ('DEBUG:ICA:testGradient:Gradient check PASSED!')
				print()
				
			result = 0;
		else:
			if self.debug:
				print ('DEBUG:ICA:testGradient:Gradient error: ', errorGrad)
				print ('DEBUG:ICA:testGradient:Gradient check FAILED!')
				print()
				
			result = -1;
			
		return result
			
	def optimizeParameters(self, X):
		'''
		Optimize for the orthonormal ICA objective, enforcing the orthonormality
		constraint. Gradient descent with a	backtracking line search is used.
		
		Arguments
		X		: data matrix in the form [input dim, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:ICA:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:ICA:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		result = 0;
		
		weightMatrix = self.unrollParameters(self.params);

		grad = self.computeGradient(self.params, X);
		
		print ('Iter', '\t', 'Cost', '\t', 't');
		 
		# Initialize some parameters for the backtracking line search
		t = 0.02;
		lastCost = 1e40;
		
		# Do iterations of gradient descent
		for iteration in range(self.maxIterations):

			grad = self.unrollParameters(grad);
			newCost = np.Inf;        
			linearDelta = np.sum(grad**2);
			
			# Perform the backtracking line search
			while 1:
				considerWeightMatrix = weightMatrix - self.alpha * grad;
				
				# Project considerWeightMatrix such that it satisfies WW^T = I
				aux1 = np.dot(considerWeightMatrix, np.transpose(considerWeightMatrix));
				aux2 = scipy.linalg.sqrtm(aux1)
				
				considerWeightMatrix = np.dot(np.linalg.inv(aux2), considerWeightMatrix);
				
				if self.debug:
					# Verify that the projection is correct
					temp = np.dot(considerWeightMatrix, np.transpose(considerWeightMatrix));
					temp = temp - np.eye(self.featureDim);
					
					if not np.sum(temp**2) < 1e-23:
						print ('WARNING:ICA:optimizeParameters: considerWeightMatrix does not satisfy WW^T = I. Check your projection again');
						result = -1;
				
				considerWeightMatrix_l = self.rollParameters(considerWeightMatrix);
				newGrad = self.computeGradient(considerWeightMatrix_l, X);
				newCost = self.computeCost(considerWeightMatrix_l, X);
				
				if newCost > (lastCost - self.alpha * t * linearDelta):
					t = 0.9 * t;
				else:
					break;
				
			lastCost = newCost;
			weightMatrix = considerWeightMatrix;

			print (iteration, '\t', round(newCost, 4), '\t', t);

			t = 1.1 * t;

			cost = newCost;
			grad = newGrad;

			if self.debug>1:
				# Visualize the learned bases as we go along    
				if np.mod(iteration, 1000) == 0:
					# Visualize the learned bases over time in different figures so 
					# we can get a feel for the slow rate of convergence
					Visualization.displayColorNetwork(np.transpose(weightMatrix)); 

		self.params = self.rollParameters(weightMatrix);
		
		return result

if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Learning orthagonal bases of coloured image patches
	# --------------------------
	
	if 1:
	  filename_data = '/home/cem/develop/UFL/data/stlSampledPatches.mat';
	else:
	  filename_data = 'C://develop//python//UFL//data//stlSampledPatches.mat';
	
	doTest 			= True;			# Test gradient computation?
	debug 			= 1;
	numPatches 		= 20000;
	imageChannels	= 3;
	patchDim		= 8;
	inputDim		= patchDim * patchDim * imageChannels;
	numFeatures	 	= 121;
	alpha			= 0.5;
	epsilon			= 1e-6;
	maxIterations	= 30000;
	costFunction	= ICA_COST_FUNCTION_ICA;

	if doTest:
		
		inputDim_test	 	= 5;
		numFeatures_test 	= 4;
		numPatches_test	 	= 10;
		costFunction		= ICA_COST_FUNCTION_ICA;
		
		sizeLayers = [inputDim_test, numFeatures_test];
	
		patches_test = np.random.rand(inputDim_test, numPatches_test);
		
		ica_test = ICA(sizeLayers, costFunction=costFunction, debug=2);
	
		print ('Checking gradient...');
		
		ica_test.testGradient(patches_test);

	# Read data from file
	data = scipy.io.loadmat(filename_data);
	patches = data['patches'][:, 0:numPatches];
	
	if debug>1:
		Visualization.displayColorNetwork(patches[:, 0:100]);
	
	# Normalize data: ZCA whiten patches
	instance_pca = PCA.PCA(inputDim, 0.99, debug);
	patches_ZCAwhite = instance_pca.doZCAWhitening(patches);
		
	if debug:
		print ('Number of samples: ', patches.shape[1]);
		
	sizeLayers = [inputDim, numFeatures];
	
	instance_ica = ICA(sizeLayers,
	                   alpha=alpha,
				       epsilon=epsilon,
	                   maxIterations=maxIterations,
					   costFunction=costFunction,
					   debug=debug);
	
	success = instance_ica.optimizeParameters(patches_ZCAwhite);
	
	# Visualize the learned bases
	weightMatrix = instance_ica.getWeights();
	Visualization.displayColorNetwork(np.transpose(weightMatrix));