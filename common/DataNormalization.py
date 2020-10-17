''' DataNormalization.py
	
	Normalization methods for data
	
	Author: Cem Karaoguz
	Date: 26.02.2015
	Version: 1.0
'''

import numpy as np

from common import AuxFunctions

def normMean(data, axis=1):
	''' 
	Removes mean
	
	data: data array/matrix, must be two dimensional
	axis: the axis along which to normalize values
	'''
	assert len(data.shape)==2, 'Data must be two dimensional, try reshaping to mx1 or 1xm array'
	assert axis in [0,1], 'Axis should be 0 or 1'
	
	axis_comp = 1-axis;
	
	m = np.resize(np.mean(data, axis), [data.shape[axis_comp], 1]);
	
	data = data - np.repeat(m, data.shape[axis], 1);
	
	return data


def normMeanStd(data, axis=1):
	''' 
	Removes mean and divides by the standard deviation
	
	data: data array/matrix, must be two dimensional
	axis: the axis along which to normalize values
	'''
	assert len(data.shape)==2, 'Data must be two dimensional, try reshaping to mx1 or 1xm array'
	assert axis in [0,1], 'Axis should be 0 or 1'
	
	axis_comp = 1-axis;
	
	s = np.resize(np.std(data, axis), [data.shape[axis_comp], 1]);
	m = np.resize(np.mean(data, axis), [data.shape[axis_comp], 1]);
	
	data = data - np.repeat(m, data.shape[axis], 1);
	data = data / np.repeat(s+0.1, data.shape[axis], 1);
	
	return data

def normZeroToOne(data, axis=1):
	''' 
	Squash data to [0.1, 0.9], useful with sigmoid as the activation function
	
	data: data array/matrix, must be two dimensional
	axis: the axis along which to normalize values
	'''
	assert len(data.shape)==2, 'Data must be two dimensional, try reshaping to mx1 or 1xm array'
	assert axis in [0,1], 'Axis should be 0 or 1'
	
	axis_comp = 1-axis;
	
	s = np.resize(np.std(data, axis), [data.shape[axis_comp], 1]);
	m = np.resize(np.mean(data, axis), [data.shape[axis_comp], 1]);
	
	# Remove DC (mean of images).
	data = data - np.repeat(m, data.shape[axis], 1);

	# Truncate to +/-3 standard deviations and scale to -1 to 1	
	pstd = np.repeat(3*s, data.shape[axis], axis);
	
	data = np.fmax(np.fmin(data, pstd), -pstd) / pstd;
	
	# Rescale from [-1,1] to [0.1,0.9]
	data = (data + 1) * 0.4 + 0.1;
	
	return data

def normL2(data, axis=1):
	''' 
	Normalizes data as x / ||x||_2
	
	data: data array/matrix, must be two dimensional
	axis: the axis along which to normalize values
	'''
	assert len(data.shape)==2, 'Data must be two dimensional, try reshaping to mx1 or 1xm array'
	assert axis in [0,1], 'Axis should be 0 or 1'
	
	m = np.sqrt(np.sum(data**2, axis) + (1e-8));
	data = AuxFunctions.doUnbalancedMatrixOperation(data, m, 'div', axis);
	
	return data
	