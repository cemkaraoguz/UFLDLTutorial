''' Linreg.py
	
	Implementation of linear regression model
	
	Author: Cem Karaoguz
	Date: 20.02.2015
	Version: 1.0
'''

import numpy as np
from scipy.optimize import minimize
import pylab as pl

from UFL.common import DataInputOutput, AuxFunctions

class Linreg:
	''' 
	Linear regression class
	'''

	def __init__(self, nParams, debug=0):
		''' 
		Initialization function of the linear regression class
		
		nParams: number of parameters (input dimensions)
		debug: debugging flag
		'''
		self.isInitialized = False;
		
		self.debug = debug;
		self.nParams = nParams;
		
		self.theta = np.random.rand(self.nParams, 1);
		
		if debug:
			print 'DEBUG:Linreg:init: initialized for nParams: ', self.nParams;
			print
		
		self.isInitialized = True;
		
	def computeCost(self, theta, X, y):
		''' 
		Computes the value of the linear regression objective function for given parameters
		(theta), data matrix (X) and corresponding labels (y) in the following form:
		
		f = 1/2 * ((h(X|theta) - y)^2)
		
		where
		
		h(X|theta) = theta'*X
		
		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data matrix in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		f		: computed cost (floating point number)
		'''
		f = 0;
		
		h = np.dot(np.transpose(theta), X);
		f = np.sum( ( h - y )**2 )/2;
		
		return f
		
	def computeGradient(self, theta, X, y):
		''' 
		Computes gradients of the linear regression objective function wrt parameters
		(theta) for a given data matrix (X) and corresponding labels (y):
		
		g = (X * (h(X|theta) - y)')
		
		where
		
		h(X|theta) = theta'*X

		Arguments
		theta	: function parameters in the form [number of parameters, 1]
		X		: data matrix in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		g		: computed gradients of parameters array in the form (number of parameters,)
		'''
		g = np.zeros(np.shape(theta));
		
		h = np.dot(np.transpose(theta), X);
		g = np.dot(X, np.transpose(h - y));
		
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
		assert self.isInitialized, 'ERROR:Linreg:testGradient: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR:Linreg:testGradient: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:Linreg:testGradient: Testing gradient computation... '
		
		result = 0
		
		grad = self.computeGradient(self.theta, X, y);
		
		numGrad = AuxFunctions.computeNumericalGradient( func=self.computeCost, params=self.theta, args=(X, y) );
		
		errorGrad = np.sqrt(np.sum((grad - numGrad)**2));
		
		if errorGrad<1e-4:
			if self.debug:
				print 'DEBUG:Linreg:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Linreg:testGradient: Gradient check PASSED!'
				print
			
			result = 0;
		else:
			if self.debug:
				print 'DEBUG:Linreg:testGradient: Gradient error: ', errorGrad
				print 'DEBUG:Linreg:testGradient: Gradient check FAILED!'
				print

			result = -1;
			
		return result
			
	def optimizeParameters(self, X, y):
		'''
		Optimizes the parameters of the linear regression model
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		y		: labels in the form [1, number of samples]
		
		Returns
		result	: result of the optimization (success or failure)
		'''
		assert self.isInitialized, 'ERROR: Linreg: optimizeParameters: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR: Linreg: optimizeParameters: Dimensions of given data do not match with the number of parameters'
		
		if self.debug: print 'DEBUG:Linreg:optimizeParameters: Optimizing paramters... '
		
		# Set optimization options
		options = {};
		
		if self.debug:
			options['disp'] = True;
			
		# Optimize the cost function
		result = minimize(fun=self.computeCost, jac=self.computeGradient, x0=self.theta, args=(X, y), options=options)
		
		# Set the new values
		self.theta = np.reshape(result.x, [self.nParams, 1]);
		
		if self.debug: print 'DEBUG:Linreg:optimizeParameters: Optimization result: ', result.message
		
		return result.success;

	def predict(self, X):
		'''
		Applies the linear regression model to the given data
		
		Arguments
		X		: data in the form [number of parameters, number of samples]
		
		Returns
		pred	: prediction vector in the form of [1, number of samples]
		'''
		assert self.isInitialized, 'ERROR: Linreg: predict: The instance is not properly initialized'
		assert X.shape[0]==self.nParams, 'ERROR: Linreg: predict: Dimensions of given data do not match with the number of parameters'
		
		pred = np.dot(np.transpose(self.theta), X);
		
		return pred
		
if __name__ == '__main__':
	
	# --------------------------
	# Example:
	# Housing price prediction
	# --------------------------
	
	housing_filename = '/home/cem/develop/UFL/data/housing.bin';
	debug = 1;
	doPlot = 1;
	
	# Read data from file
	data = DataInputOutput.loadHousingData(housing_filename);
	
	# Include a row of 1s as an additional intercept feature.
	data = np.vstack( (np.ones((1, data.shape[1])), data) );
	
	# Shuffle examples.
	data = data[:, np.random.permutation(data.shape[1])];
	
	# Split into train and test sets
	# The last row of 'data' is the median home price.
	data_train_X = data[0:-1, 0:400];
	data_train_y = data[-1:-2:-1, 0:400];
	
	data_test_X = data[0:-1, 401:-1];
	data_test_y = data[-1:-2:-1, 401:-1];
	
	inputDim = data_train_X.shape[0];
	
	linearregressor = Linreg(inputDim, debug);
	
	if debug:
		# Check if the gradient computation is OK on a smaller subset of data
		print 'Checking gradient...'
		
		linearregressor.testGradient(data_train_X[:,0:20], data_train_y[:,0:20])
	
	success = linearregressor.optimizeParameters(data_train_X, data_train_y);
	
	# Print out root-mean-squared (RMS) training error.
	actual_prices = data_train_y;
	predicted_prices = linearregressor.predict(data_train_X);
	train_rms = np.sqrt(np.mean((predicted_prices - actual_prices)**2));
	print 'RMS training error:', train_rms
	
	# Print out test RMS error
	actual_prices = data_test_y;
	predicted_prices = linearregressor.predict(data_test_X);
	train_rms = np.sqrt(np.mean((predicted_prices - actual_prices)**2));
	print 'RMS test error:', train_rms
	
	# Plot predictions on test data.
	if (doPlot):
		I = [i[0] for i in sorted(enumerate(actual_prices.flatten()), key=lambda x:x[1])];
		ap_sorted = [actual_prices[0,i] for i in I];
		pp_sorted = [predicted_prices[0,i] for i in I];
		pl.plot(ap_sorted, 'rx');
		pl.hold('on');
		pl.plot(pp_sorted, 'bx');
		pl.legend(['Actual Price', 'Predicted Price']);
		pl.xlabel('House #');
		pl.ylabel('House price ($1000s)');
		pl.show()
