''' UFL_Visualization.py
	
	Methods for visualization
	
	Author: Cem Karaoguz
	Date: 19.02.2015
	Version: 1.0
'''

import os, struct, sys
import numpy as np
import pylab as pl

def displayNetwork(A, kernelWidth=-1, kernelHeight=-1, opt_normalize=True, opt_graycolor=True, cols=-1, opt_colmajor=False):
	'''
		This function visualizes filters in matrix A. Each column of A is a
		filter. We will reshape each column into a square image and visualizes
		on each cell of the visualization panel. 
		
		opt_normalize: whether we need to normalize the filter so that all of
		them can have similar contrast. Default value is true.
		opt_graycolor: whether we use gray as the heat map. Default is true.
		cols: how many columns are there in the display. Default value is 4.
		opt_colmajor: you can switch convention to row major for A. In that
		case, each row of A is a filter. Default value is false.
	'''

	# rescale
	A = A - np.mean(A[:]);

	# compute rows, cols
	[L, M] = np.shape(A);
	if (kernelWidth<0 or kernelHeight<0):
		#sz = sqrt(L);
		w = int(np.sqrt(L));
		h = int(np.sqrt(L));
	else:
		w = kernelWidth;
		h = kernelHeight;
		
	buf = 1;
		
	if cols<=0:
		if np.floor(np.sqrt(M))**2 != M:
			n = np.ceil(np.sqrt(M));
			while np.mod(M, n)!=0 and n<1.2*np.sqrt(M):
				n = n+1;
			m = int(np.ceil(M/n));
		else:
			n = int(np.sqrt(M));
			m = int(n);
	else:
		n = int(cols);
		m = int(np.ceil(M/n));

	array = -1 * np.ones([buf+m*(w+buf), buf+n*(h+buf)]);
	
	if ~opt_graycolor:
		array = 0.1 * array;
	
	m = int(m);
	n = int(n);
	if ~opt_colmajor:
		k = 0;
		for i in range(m):
			for j in range(n):
				if (k>=M): 
					continue; 
				clim = np.max(abs(A[:,k]));
				if opt_normalize:
					array[buf+(i)*(w+buf):buf+(i)*(w+buf)+w, buf+(j)*(h+buf):buf+(j)*(h+buf)+h]  = np.reshape(A[:,k], [w, h])/clim;
				else:
					array[buf+(i)*(w+buf):buf+(i)*(w+buf)+w, buf+(j)*(h+buf):buf+(j)*(h+buf)+h] = np.reshape(A[:,k], [w, h])/np.max(abs(A[:]));
				k = k+1;
			#end j
		#end i
	else:
		k = 0;
		for j in range(n):
			for i in range(m):
				if k>=M: 
					continue; 
				clim = np.max(abs(A[:,k]));
				if opt_normalize:
					array[buf+(i)*(w+buf):buf+(i)*(w+buf)+w, buf+(j)*(h+buf):buf+(j)*(h+buf)+h]  = np.reshape(A[:,k], [w, h])/clim;
				else:
					array[buf+(i)*(w+buf):buf+(i)*(w+buf)+w, buf+(j)*(h+buf):buf+(j)*(h+buf)+h] = np.reshape(A[:,k], [w, h])/np.max(abs(A[:]));
				k = k+1;
			#end i
		#end j
	#end

	if opt_graycolor:
		#h = pl.imshow(array,'EraseMode','none',[-1 1]);
		h = pl.imshow(array, cmap='gray');
	else:
		#h = pl.imshow(array,'EraseMode','none',[-1 1]);
		h = pl.imshow(array);

	pl.axis('image')
	pl.axis('off')

	pl.show();

def displayColorNetwork(A):
	''' 
	Display receptive field(s) or basis vector(s) for image patches 
	A	: the basis, with patches as column vectors
	In case the midpoint is not set at 0, we shift it dynamically
	'''
	if np.min(A[:]) >= 0:
		A = A - np.mean(A[:]);
	
	cols = np.round(np.sqrt(A.shape[1]));

	channel_size = A.shape[0]/3;
	dim = np.sqrt(channel_size);
	dimp = dim+1;
	rows = np.ceil(A.shape[1]/cols);
	B = A[0:channel_size, :];
	C = A[channel_size:channel_size*2, :];
	D = A[2*channel_size:channel_size*3, :];
	B = B/(np.ones((B.shape[0], 1)) * np.max(np.abs(B)));
	C = C/(np.ones((C.shape[0], 1)) * np.max(np.abs(C)));
	D = D/(np.ones((D.shape[0], 1)) * np.max(np.abs(D)));
	# Initialization of the image
	I = np.ones((dim*rows+rows-1,dim*cols+cols-1,3));

	#Transfer features to this image matrix
	rows = int(rows)
	cols = int(cols)
	for i in range(rows):
		for j in range(cols):
		  
			if i*cols+j+1 > B.shape[1]:
				break
		
		# This sets the patch
		I[i*dimp:i*dimp+dim, j*dimp:j*dimp+dim, 0] = np.reshape(B[:,i*cols+j],[dim, dim]);
		I[i*dimp:i*dimp+dim, j*dimp:j*dimp+dim, 1] = np.reshape(C[:,i*cols+j],[dim, dim]);
		I[i*dimp:i*dimp+dim, j*dimp:j*dimp+dim, 2] = np.reshape(D[:,i*cols+j],[dim, dim]);

	I = I + 1;
	I = I / 2;

	pl.imshow(I);
	pl.axis('equal')
	pl.axis('off')
	pl.show();
	
if __name__ == '__main__':
	
	#W = np.random.rand(8*8, 16)
	W = np.zeros((8,8, 16))
	W[4,:,0] = 1;
	W[:,4,1] = 1;
	W = np.reshape(W, [8*8, 16])
	displayWeights(W)
	
	
