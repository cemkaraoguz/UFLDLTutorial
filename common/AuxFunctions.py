''' UFL_AuxFunction.py
	
	Auxiliary functions
	
	Author: Cem Karaoguz
	Date: 20.02.2015
	Version: 1.0
'''

import numpy as np
import sys

def computeNumericalGradient(func, params, args=()):
	'''
	Computes numerical gradients of a function
	'''
	# Initialize numgrad with zeros
	numgrad = np.zeros(np.shape(params));
	
	eps = 0.0001;
	for i in range(params.size):
		theta_plus = params.copy();
		theta_minus = params.copy();
		theta_plus[i] = theta_plus[i] + eps;
		theta_minus[i] = theta_minus[i] - eps;
		
		f_plus = func(theta_plus, *args);
		f_minus = func(theta_minus, *args);
		
		numgrad[i] = (f_plus - f_minus) / (2*eps);
		
	return numgrad.flatten()

def sigmoid(x):
	'''
	Computes sigmoid response
	'''
	return 1.0 / (1.0 + np.exp(-x));
	
def doUnbalancedMatrixOperation(x, y, operation, axis=1):
	'''
	Computes a given matrix operation between two operands where
	the second operand needs to be filled (repeated) to the size of the
	first operand
	
	x			: first operand
	y			: second operand
	operation	: operation, possible values ['add', 'sub', 'mul', 'div']
	axis		: axis of operation, possible values [0, 1]
	'''
	oplist = ['add', 'sub', 'mul', 'div'];
	
	if operation not in oplist:
		print ('ERROR: operation not recognized, permitted operations: ')
		print (oplist)
		sys.exit()
	
	assert len(np.shape(x))==2, 'First operand must be two dimensional'
	assert len(np.shape(y)) in [1,2], 'Second operand must be one or two dimensional'
	assert axis in [0,1], 'Axis should be 0 or 1'
	
	# Reshape the second operand as 2 dimensional vector
	if len(np.shape(y))==1:
		if axis==0:
			resizearray = [1, len(y)]
		else:
			resizearray = [len(y), 1]
		
	aux1 = np.resize(y, resizearray);
	aux2 = np.repeat(aux1, np.shape(x)[axis], axis)
	
	if operation==oplist[0]:
		return x + aux1
	elif operation==oplist[1]:
		return x - aux1
	elif operation==oplist[2]:
		return x * aux1
	elif operation==oplist[3]:
		return x / aux1

def checkNetworkParameters(params, topology):
	'''
	Checks if the structure of given parameters is consistent with a given network
	topology
	
	Arguments
	params		: List of parameters. The length of list corresponds to layer size, contents of the list are the parameter matrices/vectors
	topology	: List of tuples. The length of list corresponds to layer size, contents of the list is the layer topology i.e. (input dims, output dims)
	
	Returns
	result		: True if the parameter structure is consistent with the topology, false otherwise
	'''
	
	result = True;
	
	# First check the number of layers 
	nLayers = len(topology);
	result = result and len(params)==nLayers;
	# Check the topology
	for i in range(nLayers):
		for j in range(len(np.shape(topology[i]))):
			result = result and topology[i][j] == np.shape(params[i])[j];
	
	return result