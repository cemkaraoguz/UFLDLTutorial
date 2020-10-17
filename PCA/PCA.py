''' PCA.py
	
	Implementation of Principal Component Analysis
	and other related algorithms
	
	Author: Cem Karaoguz
	Date: 12.03.2015
	Version: 1.0
'''

import sys
import numpy as np
import pylab as pl
from scipy.optimize import minimize

from common import DataInputOutput, DataNormalization, AuxFunctions, Visualization

class PCA:
	''' 
	Principal Component Analysis
	'''

	def __init__(self,
				 inputDim,
				 tolerance=0.99,
				 epsilon=0,
				 debug=0):
		''' 
		Initialization function of the PCA class
		
		Arguments
		inputDim	: dimension of the input data
		tolerance	: dimension reduction will retain retain at least the tolerance of the variance, default is 0.99
		epsilon		: epsilon for covariance matrix regularisation, default is 0
		debug		: debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.inputDim = inputDim;
		self.tolerance = tolerance;
		self.epsilon = epsilon;
		
		self.U = [];
		self.V = [];
		self.S = [];
		
		if debug:
			print ('DEBUG:PCA:init: initialized for inputDim: ', inputDim);
			print ('DEBUG:PCA:init: initialized for tolerance: ', tolerance);
			print()
		
		self.isInitialized = True;

	def normalizeData(self, X):
		''' 
		Checks if data is normalized. If not, performs zero-mean normalization.
		
		Arguments
		X 		: data matrix in the form [number of dimensions, number of samples].
		
		Returns
		X_norm	: data matrix normalized by mean subtraction
		'''
		m = np.resize(np.mean(X, 1), [X.shape[0], 1]);
		if np.sum(np.abs(m))>1e-5:
			X_norm = X - np.repeat(m, X.shape[1], 1);
		else:
			X_norm = X;
			
		return X_norm
		
	def computeEigenbases(self, X):
		''' 
		Computes the eigenspace of the covariance matrix of given data (X) using singular value decomposition:
		
		U, S, V = svd(X)
		
		U is composed of the principal components of the data.
		Performs zero-mean normalization on data if not done beforehand
		
		Arguments
		X 	: data, should be in the form [number of dimensions, number of samples].
		'''
		assert self.isInitialized, 'ERROR:PCA:computeEigenbases: Object was not initialized properly'
		assert len(X.shape)==2, 'ERROR:PCA:computeEigenbases: Data should be in the form [number of parameters, number of samples]'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:computeEigenbases: Data dimension does not match with the internal parameter'
		
		# Zero-mean the data (by row)
		X_norm = self.normalizeData(X);
		
		# Use PCA to obtain xRot, the matrix in which the data is expressed 
		# with respect to the eigenbasis of sigma, which is the matrix U.
		
		if self.debug: print ('DEBUG:PCA:computeEigenbases: computing SVD...');
		
		#sigma = np.dot(X_norm, np.transpose(X_norm)) / X_norm.shape[1];
		sigma = np.dot(X_norm, np.transpose(X_norm));
		[U,S,V] = np.linalg.svd(sigma);
		
		self.U = U;
		self.V = V;
		self.S = S;
		
		if self.debug>1:
			# Check the implementation of PCA
			# The covariance matrix for the data expressed with respect to the basis U
			# should be a diagonal matrix with non-zero entries only along the main
			# diagonal.
			X_Rot = np.dot(np.transpose(U), X_norm);
		
			#covar = np.dot(X_Rot, np.transpose(X_Rot)) / X_Rot.shape[1];
			covar = np.dot(X_Rot, np.transpose(X_Rot));
		
			# Visualise the covariance matrix. You should see a line across the
			# diagonal against a blue background.
			pl.figure();
			pl.title("Covariance matrix of the data projected to eigenbases");
			pl.imshow(covar);
			pl.show();
		
	def computeComponentNumber(self):
		''' 
		Computes the number of components to retain a given variance ratio when data is projected onto its principal components.
		Eigenspace should be computed beforehand.
		
		Returns
		k	: number of principal components to use to retain given variance ratio
		'''
		assert self.isInitialized, 'ERROR:PCA:computeComponentNumber: Object was not initialized properly'
		assert len(self.S)>0, 'ERROR:PCA:computeComponentNumber: Eigenbases were not yet computed'
		assert len(self.U)>0, 'ERROR:PCA:computeComponentNumber: Eigenbases were not yet computed'
		
		sum_ev_all = np.sum(self.S);
		sum_ev = 0.0;
		k = self.U.shape[1];
		for i in range(self.U.shape[1]):
			sum_ev = sum_ev + self.S[i];
			if(sum_ev/sum_ev_all)>=self.tolerance:
				k = i;
				break
				
		if self.debug: print ('DEBUG:PCA:computeComponentNumber: k = ', k)
		
		return k;
	
	def projectDataToEigenbases(self, X, nDims, doBackProject=False):
		''' 
		Projects data (X) to a sub-space composed of its first (nDims) principal components
		Performs zero-mean normalization on data if not done beforehand
		Reduces the dimensions of data
		
		Arguments
		X				: data in the form [number of dimensions, number of samples]
		nDims			: reduced dimension i.e. number of components to be used
		doBackProject 	: whether a back projection of data to the original dimension shall be done, default is False
		
		Returns
		X_Red			: data (X) projected to the principal component space
		X_Hat			: reduced data (X_Red) back-projected to the original space, empty if doBackProject=False
		'''
		assert self.isInitialized, 'ERROR:PCA:projectDataToEigenbases: Object was not initialized properly'
		assert nDims >0, 'ERROR:PCA:projectDataToEigenbases: Number of reduced dimensions must be greater than zero'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:projectDataToEigenbases: Data dimension does not match with the internal parameter'
		assert len(X.shape)==2, 'ERROR:PCA:projectDataToEigenbases: Data should be in the form [number of parameters, number of samples]'
		
		# Reduced dimensions must be smaller than the actual dimensions
		nDims = min(nDims, self.U.shape[1]);
		
		# Check if data is normalized
		X_norm = self.normalizeData(X);
		
		# Check if eigenspace is already computed
		if len(self.U)==0:
			
			if self.debug: print ('DEBUG:PCA:projectDataToEigenbases: Computing eigenspace');
			
			self.computeEigenbases(X_norm);
			
		X_Red = np.dot(np.transpose(self.U[:,0:nDims]), X_norm);
		
		if doBackProject:
		
			if self.debug: print ('DEBUG:PCA:projectDataToEigenbases: Backprojecting data');
			
			X_Hat = np.dot(self.U[:,0:nDims], X_Red);
			
			if self.debug>1:
				# Visualise the data, and compare it to the raw data
				randsel = np.random.randint(X.shape[1], size=200);
				pl.figure();
				pl.title("PCA reconstruction - dimensions: " + str(nDims));
				Visualization.displayNetwork(X_Hat[:,randsel]);
		else:
			X_Hat = [];
			
		return X_Red, X_Hat
		
	def computePCAWhiteningMatrix(self, X):
		''' 
		Computes PCA whitening matrix following:
		
		PCAWhite = 1/sqrt(S + epsilon) * U'
		
		where
		
		U, S, V = svd(X),
		epsilon is a small number
		
		Performs zero-mean normalization on data if not done beforehand
		Performs eigenbase computation if not done beforehand
		
		Arguments
		X			: data in the form [number of dimensions, number of samples]
		
		Returns
		PCAwhite	: PCA whitening matrix
		'''
		assert self.isInitialized, 'ERROR:PCA:computePCAWhiteningMatrix: Object was not initialized properly'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:computePCAWhiteningMatrix: Data dimension does not match with the internal parameter'
		assert len(X.shape)==2, 'ERROR:PCA:computePCAWhiteningMatrix: Data should be in the form [number of parameters, number of samples]'
		
		# Check if data is normalized
		X_norm = self.normalizeData(X);

		# Check if eigenspace is already computed
		if len(self.U)==0:
			
			if self.debug: print ('DEBUG:PCA:doPCAWhitening: Computing eigenbases');
			
			self.computeEigenbases(X_norm);
			
		PCAwhite = np.dot(np.diag(1./np.sqrt(self.S + self.epsilon)), np.transpose(self.U));
		
		return PCAwhite
		
	def computeZCAWhiteningMatrix(self, X):
		''' 
		Computes ZCA whitening matrix following:
		
		ZCAWhite = U * 1/sqrt(S + epsilon) * U'
		
		where
		
		U, S, V = svd(X),
		epsilon is a small number
		
		Performs zero-mean normalization on data if not done beforehand
		Performs eigenbase computation if not done beforehand
		
		Arguments
		X			: data in the form [number of dimensions, number of samples]
		
		Returns
		ZCAwhite	: ZCA whitening matrix
		'''
		assert self.isInitialized, 'ERROR:PCA:computeZCAWhiteningMatrix: Object was not initialized properly'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:computeZCAWhiteningMatrix: Data dimension does not match with the internal parameter'
		assert len(X.shape)==2, 'ERROR:PCA:computeZCAWhiteningMatrix: Data should be in the form [number of parameters, number of samples]'

		# Check if data is normalized
		X_norm = self.normalizeData(X);
		
		# Check if eigenspace is already computed
		if len(self.U)==0:
			
			if self.debug: print ('DEBUG:PCA:doPCAWhitening: Computing eigenbases');
			
			self.computeEigenbases(X_norm);
			
		PCAwhite = self.computePCAWhiteningMatrix(X_norm);
		ZCAWhite = np.dot(self.U, PCAwhite);
		
		return ZCAWhite

	def doPCAWhitening(self, X):
		''' 
		Performs PCA whitening following:
		
		X_PCAWhite = 1/sqrt(S + epsilon) * U' * X
		
		Arguments
		X			: data in the form [number of dimensions, number of samples]
		
		Returns
		X_PCAwhite	: PCA whitened data
		'''
		assert self.isInitialized, 'ERROR:PCA:doPCAWhitening: Object was not initialized properly'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:doPCAWhitening: Data dimension does not match with the internal parameter'
		assert len(X.shape)==2, 'ERROR:PCA:doPCAWhitening: Data should be in the form [number of parameters, number of samples]'
		
		# Check if data is normalized
		X_norm = self.normalizeData(X);
		
		PCAwhite = self.computePCAWhiteningMatrix(X_norm);
		
		X_PCAwhite = np.dot(PCAwhite, X_norm);
		
		if self.debug>1:
			# Check the implementation of PCA whitening with and without regularisation. 
			# PCA whitening without regularisation results a covariance matrix 
			# that is equal to the identity matrix. PCA whitening with regularisation
			# results in a covariance matrix with diagonal entries starting close to 
			# 1 and gradually becoming smaller.

			covar = np.dot(X_PCAwhite, np.transpose(X_PCAwhite)) / X_PCAwhite.shape[1];

			# Visualise the covariance matrix. You should see a red line across the
			# diagonal against a blue background.
			pl.figure();
			pl.title("Covariance matrix of the data after PCA Whitening");
			pl.imshow(covar);
			pl.show();
			
		return X_PCAwhite
		
	def doZCAWhitening(self, X):
		''' 
		Performs ZCA whitening following:
		
		Z_PCAWhite = U * 1/sqrt(S + epsilon) * U' * X
		
		Arguments
		X			: data in the form [number of dimensions, number of samples]
		
		Returns
		X_ZCAwhite	: ZCA whitened data
		'''
		assert self.isInitialized, 'ERROR:PCA:doZCAWhitening: Object was not initialized properly'
		assert X.shape[0]==self.inputDim, 'ERROR:PCA:doZCAWhitening: Data dimension does not match with the internal parameter'
		assert len(X.shape)==2, 'ERROR:PCA:doZCAWhitening: Data should be in the form [number of parameters, number of samples]'

		# Check if data is normalized
		X_norm = self.normalizeData(X);
		
		ZCAwhite = self.computeZCAWhiteningMatrix(X_norm);
		
		X_ZCAWhite = np.dot(ZCAwhite, X_norm);
		
		if self.debug>1:
			# Visualise the data, and compare it to the raw data.
			# You should observe that the whitened images have enhanced edges.
			randsel = np.random.randint(X.shape[1], size=200);
			pl.figure();
			pl.title("Data after ZCA Whitening");
			Visualization.displayNetwork(X_ZCAWhite[:,randsel]);
			
		return X_ZCAWhite
		
if __name__ == '__main__':
	
	if 1:
	  mnist_img_filename_train = '/home/cem/develop/UFL/data/train-images-idx3-ubyte';
	  mnist_img_filename_test = '/home/cem/develop/UFL/data/t10k-images-idx3-ubyte';
	else:
	  mnist_img_filename_train = 'C://develop//python//UFL//data//train-images-idx3-ubyte';
	  mnist_img_filename_test = 'C://develop//python//UFL//data//t10k-images-idx3-ubyte';
	
	
	debug = 2;
	nSamples_max_train = 20000;
	nSamples_max_test = 10000;
	tolerance = 0.99;
	
	# Read data from file
	images_train = DataInputOutput.loadMNISTImages(mnist_img_filename_train, nSamples_max_train);
	images_test = DataInputOutput.loadMNISTImages(mnist_img_filename_test, nSamples_max_test);
	
	inputDim_train, nSamples_train = np.shape(images_train);
	inputDim_test , nSamples_test  = np.shape(images_test);
	
	# Normalize data 
	images_train = images_train / 255.0;
	images_test = images_test / 255.0;
	images_train_norm = DataNormalization.normMean(images_train);
	images_test_norm = DataNormalization.normMean(images_test);
	
	# Random indices for samples to visualize
	randsel = np.random.randint(nSamples_train, size=200);
	
	if debug>1:
		pl.figure();
		pl.title("Random samples from input data")
		Visualization.displayNetwork(images_train[:,randsel]);
	
	if debug:
		print ('Number of training samples: ', nSamples_train)
		print ('Number of test samples: ', nSamples_test)
	
	# Zero-mean the data (by row)
	instance_pca = PCA(inputDim_train, tolerance, debug)
	
	instance_pca.computeEigenbases(images_train_norm);
	
	# Find k, the number of components to retain at least tolerance% of the variance.
	nDim_red = instance_pca.computeComponentNumber();
	
	if 0:
		# Now k is found, the dimension of the data can be reduced by
		# discarding the remaining dimensions. 
		xRed, xHat = instance_pca.projectDataToEigenbases(images_train_norm, nDim_red, doBackProject=True)
		
		# Implement PCA with whitening and regularisation
		xPCAwhite = instance_pca.doPCAWhitening(images_train_norm);
		
		# Implement ZCA whitening
		xZCAwhite = instance_pca.doZCAWhitening(images_train_norm);
	
	# Try on the test data set
	if 1:
		if debug>1:
			pl.figure();
			pl.title("Random samples from test data set")
			Visualization.displayNetwork(images_train[:,randsel]);
		
		# Reduce dimensions of test data set using eigenbases of training data set
		xRed, xHat = instance_pca.projectDataToEigenbases(images_test_norm, nDim_red, doBackProject=True)
		
		# Implement PCA with whitening and regularisation
		xPCAwhite = instance_pca.doPCAWhitening(images_test_norm);
		
		# Implement ZCA whitening
		xZCAwhite = instance_pca.doZCAWhitening(images_test_norm);
		