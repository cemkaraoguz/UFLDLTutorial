''' SparseCoding.py
	
	Implementation of Sparse Coding
	
	Author: Cem Karaoguz
	Date: 16.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize
import scipy.io

from UFL.common import DataInputOutput, DataNormalization, AuxFunctions, Visualization

SparseCoding_ACTIVATION_FUNCTIONS = ['sigmoid']
SparseCoding_ACTFUN_SIGMOID = 0;

class SparseCoding:
	''' 
	Sparse Coding
	'''

	def __init__(self,
	             inputDim,
				 featureDim,
				 poolDim = 0,
				 lambd = 5e-5,
				 epsilon = 1e-5,
				 gamma = 1e-2,
				 batchNumPatches = 2000,
				 debug=0):
		''' 
		Initialization function of the Sparse Coding class
		
		Arguments
		inputDim		: input dimensions 
		featureDim		: features dimensions 
		poolDim			: dimension of the grouping region (poolDim x poolDim) for topographic sparse coding, default is 0 (no pooling)
		lambd			: L1-regularisation parameter (on features), default is 5e-5
		epsilon			: L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon), default is 1e-5
		gamma			: L2-regularisation parameter (on basis), default is 1e-2
		batchNumPatches	: number of patches per batch, default is 2000
		debug			: debugging flag
		'''
		self.isInitialized = False;
		
		assert inputDim>0, 'ERROR:SparseCoding:init: inputDim should be >0'
		assert featureDim>0, 'ERROR:SparseCoding:init: featureDim should be >0'
		assert np.floor(np.sqrt(featureDim))**2==featureDim, 'featureDim should be a perfect square';
		assert poolDim>=0, 'ERROR:SparseCoding:init: poolDim should be >=0'
		assert lambd>=0, 'ERROR:SparseCoding:init: lambda should be >=0'
		assert epsilon>=0, 'ERROR:SparseCoding:init: epsilon should be >=0'
		assert gamma>=0, 'ERROR:SparseCoding:init: gamma should be >=0'
		assert batchNumPatches>0, 'ERROR:SparseCoding:init: batchNumPatches should be >0'
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.featureDim = featureDim;
		self.poolDim = poolDim;
		self.lambd = lambd;
		self.epsilon = epsilon;
		self.gamma = gamma;
		self.batchNumPatches = batchNumPatches;
		
		self.maxIterations = 2000;
		
		self.weights_vec  = np.random.rand( self.inputDim * self.featureDim ) * 0.005;
		self.features_vec = np.random.rand( self.featureDim * self.batchNumPatches ) * 0.005;
		
		if self.poolDim==0:
			self.groupMatrix = np.eye(self.featureDim);
		else:
			donutDim = int(np.floor(np.sqrt(self.featureDim)));
			
			self.groupMatrix = np.zeros((self.featureDim, donutDim, donutDim));

			groupNum = 0;
			for row in range(donutDim):
				for col in range(donutDim):
					self.groupMatrix[groupNum, 0:poolDim, 0:poolDim] = 1;
					groupNum = groupNum + 1;
					self.groupMatrix = np.roll(self.groupMatrix, -1, axis=2);
				
				self.groupMatrix = np.roll(self.groupMatrix, -1, axis=1);
			
			self.groupMatrix = np.reshape(self.groupMatrix, [self.featureDim, self.featureDim]);
		
		if debug:
			print 'DEBUG:SparseCoding:init: initialized for inputDim: ', self.inputDim;
			print 'DEBUG:SparseCoding:init: initialized for : featureDim:', self.featureDim;
			print 'DEBUG:SparseCoding:init: initialized for : poolDim:', self.poolDim;
			print 'DEBUG:SparseCoding:init: initialized for : lambd:', self.lambd;
			print 'DEBUG:SparseCoding:init: initialized for : epsilon:', self.epsilon;
			print 'DEBUG:SparseCoding:init: initialized for : gamma:', self.gamma;
			print 'DEBUG:SparseCoding:init: initialized for : batchNumPatches', self.batchNumPatches;
			print
		
		self.isInitialized = True;
		
	def rollParameters(self, theta):
		''' 
		Converts parameters in matrix for to vector
		
		Arguments
		theta	: parameters in matrix format
		
		Returns
		theta	: parameter in vector format
		'''
		return theta.flatten();
		
	def unrollParameters(self, theta_w, theta_f):
		''' 
		Converts the vectorized parameters into matrix
		
		Arguments
		theta_w		: weight parameters in vector format
		theta_f		: feature parameters in vector format
		
		Returns
		weights		: weight parameters in matrix format
		features	: feature parameters in matrix format
		'''
		assert np.size(theta_w)==self.inputDim*self.featureDim, 'ERROR:SparseCoding:unrollParameters: Bad dimensions for weight matrix'
		assert np.size(theta_f)==self.featureDim*self.batchNumPatches, 'ERROR:SparseCoding:unrollParameters: Bad dimensions for feature matrix'
		
		weights  = np.reshape( theta_w, [self.inputDim, self.featureDim] );
		features = np.reshape( theta_f, [self.featureDim, self.batchNumPatches] );
		
		return weights, features
	
	def getWeightMatrix(self):
		''' 
		Returns the weight matrix
		'''
		weightMatrix, featureMatrix = self.unrollParameters(self.weights_vec, self.features_vec);
		
		return weightMatrix
		
	def computeWeightCost(self, weights, features, X):
		''' 
		For a given weight matrix, feature matrix and data matrix, computes the value of the 
		Sparse Coding objective function:
		
		f = f_f + f_s + f_r
		
		Here, f_f is the fidelity term. f_s is the sparsity term and f_r is the regularization term:
		
		f_f = || A*s - x ||_2,
		f_s = lambda * sqrt(s^2 + epsilon)
		f_r = gamma * || A ||_2
		
		where A is the weight matrix, s is the feature matrix, 
		||.||_k denotes Lk norm: sum_i(|(x_i)^k|)^(1/k)
		
		Computes the same value as computeFeatureCost function, implemented due to the argument order
		requirement of the third party optimization library
		
		Arguments
		weights		: weight parameters in vector format
		features	: feature parameters in vector format
		X			: data in the form [number of parameters, number of samples]
		
		Returns
		f			: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:computeWeightCost: The instance is not properly initialized'
		
		weightMatrix, featureMatrix = self.unrollParameters(weights, features);
		
		f = 0;
		
		numExamples = X.shape[1];

		cost_fidelity = (1.0/numExamples) * np.sum(1.0 * (np.dot(weightMatrix, featureMatrix) - X)**2);
		cost_regularization = (self.gamma/2.0) * (np.sum(weightMatrix**2));
		cost_sparsity = (1.0/numExamples) * self.lambd * np.sum(np.sqrt(np.dot(self.groupMatrix, (featureMatrix**2)) + self.epsilon));
		
		f = cost_fidelity + cost_sparsity + cost_regularization;
		
		return f
		
	def computeWeightGradient(self, weights, features, X):
		''' 
		Computes gradients of the Sparse Coding objective function with respect to weight matrix:
		
		g = (2.0/numExamples) * (A*s - X) * s' + gamma * A;

		where A is the weight matrix, s is the feature matrix.
		
		Arguments
		weights		: weight parameters in vector format
		features	: feature parameters in vector format
		X			: data matrix in the form [input dim, number of samples]
		
		Returns
		grad		: gradients of weights in rolled form
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:computeWeightGradient: The instance is not properly initialized'
		
		numExamples = X.shape[1];
		
		weightMatrix, featureMatrix = self.unrollParameters(weights, features);
		
		grad = 2.0 * (1.0/numExamples) * np.dot((np.dot(weightMatrix, featureMatrix) - X), np.transpose(featureMatrix)) + self.gamma * weightMatrix;
		
		return self.rollParameters(grad);
	
	def computeFeatureCost(self, features, weights, X):
		''' 
		For a given weight matrix, feature matrix and data matrix, computes the value of the 
		Sparse Coding objective function:
		
		f = f_f + f_s + f_r
		
		Here, f_f is the fidelity term. f_s is the sparsity term and f_r is the regularization term:
		
		f_f = || A*s - x ||_2,
		f_s = lambda * sqrt(s^2 + epsilon)
		f_r = gamma * || A ||_2
		
		where A is the weight matrix, s is the feature matrix, 
		||.||_k denotes Lk norm: sum_i(|(x_i)^k|)^(1/k)
		Further extension on f_s was implemented to achieve topological sparse coding:
		
		f_s = lambda * sum(sqrt(V*s*s' + epsilon))
		
		where V is the [featureDim, featureDim] sized grouping matrix such that rth row of V indicates 
		which features are grouped in the rth group.
		
		Computes the same value as computeFeatureCost function, implemented due to the argument order
		requirement of the third party optimization library
		
		Arguments
		features	: feature parameters in vector format
		weights		: weight parameters in vector format
		X			: data in the form [number of parameters, number of samples]
		
		Returns
		f			: computed cost (floating point number)
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:computeFeatureCost: The instance is not properly initialized'
		
		weightMatrix, featureMatrix = self.unrollParameters(weights, features);
		
		f = 0;
		
		numExamples = X.shape[1];

		cost_fidelity = (1.0/numExamples) * np.sum(1.0 * (np.dot(weightMatrix, featureMatrix) - X)**2);
		cost_sparsity = self.lambd * np.sum(np.sqrt(np.dot(self.groupMatrix, (featureMatrix**2)) + self.epsilon));
		cost_regularization = (self.gamma/2.0) * (np.sum(weightMatrix**2));
		
		f = cost_fidelity + cost_sparsity + cost_regularization;
		
		return f
		
	def computeFeatureGradient(self, features, weights, X):
		''' 
		Computes gradients of the Sparse Coding objective function with respect to feature matrix:
		
		g = (1.0/numExamples) * 2 * A' * (A*s - X) + lambda * V' * ((V * s^2) + epsilon)^(-0.5) * s;
		
		where A is the weight matrix, s is the feature matrix, V is the group matrix.
		
		Arguments
		weights		: weight parameters in vector format
		features	: feature parameters in vector format
		X			: data matrix in the form [input dim, number of samples]
		
		Returns
		grad		: gradients of features in rolled form
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:computeFeatureGradient: The instance is not properly initialized'
		
		numExamples = X.shape[1];
		
		weightMatrix, featureMatrix = self.unrollParameters(weights, features);
		
		aux1 = (1.0/numExamples) * 2 * np.dot(np.transpose(weightMatrix), (np.dot(weightMatrix, featureMatrix) - X))
		aux2 = (np.dot(self.groupMatrix, (featureMatrix**2)) + self.epsilon)**(-0.5);
		#aux2 = np.linalg.inv(scipy.linalg.sqrtm(np.dot(self.groupMatrix, (featureMatrix**2)) + self.epsilon));
		
		grad = aux1 + self.lambd * np.dot(np.transpose(self.groupMatrix), (aux2 * featureMatrix));
		
		return self.rollParameters(grad);
		
	def testGradient(self, X):
		'''
		Tests the analytical gradient computation by comparing it with the numerical gradients

		Arguments
		X		: data matrix the form [input dim., number of samples]
		
		Returns
		result	: 0 if passed, -1 if failed
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:testGradient: The instance is not properly initialized'
		
		if self.debug: print  'DEBUG:SparseCoding:testGradient:Checking weight gradient...'
		
		result = 0;
		
		grad = self.computeWeightGradient(self.weights_vec, self.features_vec, X);

		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeWeightCost, params=self.weights_vec, args=(self.features_vec, X) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:SparseCoding:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseCoding:testGradient:Gradient check PASSED!'
				print
		else:
			if self.debug:
				print 'DEBUG:SparseCoding:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseCoding:testGradient:Gradient check FAILED!'
				print
			
			result = -1;

		if self.debug: print  'DEBUG:SparseCoding:testGradient:Checking feature gradient...'
		
		grad = self.computeFeatureGradient(self.features_vec, self.weights_vec, X);

		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeFeatureCost, params=self.features_vec, args=(self.weights_vec, X) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:SparseCoding:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseCoding:testGradient:Gradient check PASSED!'
				print
		else:
			if self.debug:
				print 'DEBUG:SparseCoding:testGradient:Gradient error: ', errorGrad
				print 'DEBUG:SparseCoding:testGradient:Gradient check FAILED!'
				print
				
			result = -1;
			
		return result
			
	def optimizeParameters(self, X):
		'''
		Optimizes the parameters of the Sparse Coding model. Optimization executes two steps alternatingly:
		
		- Initialize weight matrix randomly
		- Repeat until convergence:
			1) Find feature matrix that minimizes cost for weight matrix from the previous step
			2) Solve for weight matrix that minimizes the cost for the feature matrix from the previous step
			
		Each iteration selects and works on a mini batch from the whole data set to increase the rate of 
		convergence and speed up the algorithm.
		
		Good initialization of the feature matrix is also important, this is done by selecting the feature 
		matrix as:
		
			1) s = A'*x 
			2) s_{r,c} = s_{r,c} / ||A_c||
		
		where s is the feature matrix, A is the weight matrix and x is the mini-batch data matrix. Step 1 initializes
		the feature matrix to achieve Ws~x, step 2 normalizes the matrix to keep the values small (hence, reduces
		the sparsity cost).

		Arguments
		X		: data in the form [input dim., number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR:SparseCoding:optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.inputDim, 'ERROR:SparseCoding:optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:SparseCoding:optimizeParameters:Optimizing parameters...'
		
		weightMatrix, featureMatrix = self.unrollParameters(self.weights_vec, self.features_vec);
		
		# Set optimization options
		#method = 'L-BFGS-B'
		method = 'CG' 
		options = {};
		options['maxiter'] = 300;
		options['disp'] = False;
		
		# Initial batch
		indices = np.random.permutation(numPatches);
		indices = indices[0:batchNumPatches];
		batchPatches = X[:, indices];

		if self.debug: print 'Iter', '\t', 'fObj', '\t', 'fRes', '\t', 'fSpar', '\t', 'fWeight';

		for iteration in range(self.maxIterations):
			error = np.dot(weightMatrix, featureMatrix) - batchPatches;
			error = np.sum(error**2) / (batchNumPatches * 1.0);

			fResidue = error;

			R = np.dot(self.groupMatrix, (featureMatrix**2));
			R = np.sqrt(R + self.epsilon);    
			fSparsity = self.lambd * np.sum(R);    

			fWeight = self.gamma * np.sum(weightMatrix**2);

			if self.debug: print iteration, '\t', np.round(fResidue+fSparsity+fWeight,4), '\t', np.round(fResidue,4), '\t', np.round(fSparsity,4), '\t', np.round(fWeight,4)

			# Select a new batch
			indices = np.random.permutation(numPatches);
			indices = indices[0:batchNumPatches];
			batchPatches = X[:, indices];

			# Reinitialize featureMatrix with respect to the new batch
			featureMatrix = np.dot(np.transpose(weightMatrix), batchPatches);
			normWM = np.transpose(np.sum(weightMatrix**2, 0));
			featureMatrix = featureMatrix/np.repeat(np.reshape(normWM, [len(normWM), 1]), featureMatrix.shape[1], 1);

			# Optimize for feature matrix
			options['maxiter'] = 20;
			
			features_vec = self.rollParameters(featureMatrix);
			weights_vec = self.rollParameters(weightMatrix);
			
			result = minimize(fun=self.computeFeatureCost, jac=self.computeFeatureGradient, x0=self.features_vec, args=(weights_vec, batchPatches), method=method, options=options)
			
			self.features_vec = result.x
			weightMatrix, featureMatrix = self.unrollParameters(weights_vec, self.features_vec);

			# Optimize for weight matrix  
			weightMatrix = np.zeros((self.inputDim, self.featureDim));
			if 1:
				# Analytic solution
				aux1 = np.dot(batchPatches, np.transpose(featureMatrix));
				aux2 = self.gamma * np.dot(batchNumPatches, np.eye(featureMatrix.shape[0]));
				aux3 = np.dot(featureMatrix, np.transpose(featureMatrix));
				
				weightMatrix = np.dot(aux1,  np.linalg.inv(aux2 +  aux3));
				
			else:
				# Gradient descent
				result = minimize(fun=self.computeWeightCost, jac=self.computeWeightGradient, x0=weights_vec, args=(self.features_vec, batchPatches), method=method, options=options)
				
				self.weights_vec = result.x
				weightMatrix, featureMatrix = self.unrollParameters(self.weights_vec, self.features_vec);
				
			if self.debug:
				grad = self.computeWeightGradient(weightMatrix, featureMatrix, batchPatches);
				if (np.linalg.norm(grad) < 1e-12):
					print 'DEBUG:SparseCoding:optimizeParameters:Weight gradient is okay.'
				else:
					print 'WARNING:SparseCoding:optimizeParameters:Weight gradient =', np.round(np.linalg.norm(grad),4), ' should be close to 0.'
		
		return result.success;
		
if __name__ == '__main__':
	
	if 1:
	  filename_data = '/home/cem//develop/UFL/data/IMAGES.mat';
	else:
	  filename_data = 'C://develop//python//UFL//data//IMAGES.mat';
	
	debug 			= 2;
	numPatches 		= 20000;
	batchNumPatches = 2000;		# number of patches per batch
	patchWidth 		= 8;
	patchHeight 	= 8;
	imChannels	 	= 1;
	inputDim 		= patchWidth * patchHeight * imChannels;
	numFeatures 	= 121;
	poolDim 		= 3;
	poolDim 		= 0;
	lambd 			= 5e-5;		# L1-regularisation parameter (on features)
	epsilon 		= 1e-5; 	# L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
	gamma 			= 1e-2;   	# L2-regularisation parameter (on basis)

	# Read data from file
	data = scipy.io.loadmat(filename_data);
	
	patches = DataInputOutput.samplePatches(data['IMAGES'], patchWidth, patchHeight, numPatches);
	
	patches = DataNormalization.normMeanStd(patches);
	
	if debug>1:
		#Visualization.displayNetwork(patches[:, 0:64]);

		featureDim_test = 4;
		numPatches_test = 6;
		
		sc_test = SparseCoding(inputDim,
		                       featureDim_test,
							   poolDim = poolDim,
							   batchNumPatches=numPatches_test,
					           debug = debug);
							   
		print 'Checking gradient:';
		
		sc_test.testGradient(patches[:,0:numPatches_test]);
		
	sc = SparseCoding(inputDim,
	                  numFeatures,
	    			  poolDim = poolDim,
					  lambd = lambd,
					  epsilon = epsilon,
					  gamma = gamma,
					  batchNumPatches = batchNumPatches,
					  debug = debug);
	
	success = sc.optimizeParameters(patches);
	
	# Visualize learned basis
	weightMatrix = sc.getWeightMatrix();
	Visualization.displayNetwork(weightMatrix);           
	