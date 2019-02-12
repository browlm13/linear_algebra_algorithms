#!/usr/bin/env python3

"""
	Forward / Backward Substitution - Row / Column Oriented


	Column / Row change rules:
	
	Row:

	for i in range(full row, forward or backward):

		for j in (oposite):
			same expression

		outer loop expression bellow (same expression)
	
	Column:

	for j in range(full column, forward or backward):

		outer loop expression above (same expression)

		for i in (oposite):
			same expression

	oposites:

		(0,1,...,k) <-> (k+1,k+2,n/m)

"""

import numpy as np

#
# Lx = b
#

def row_oriented_forward_substitution(L,b):

	for i in range(L.shape[0]): 
		for j in range(i):
			b[i] -= L[i,j]*b[j]
		b[i] = b[i]/L[i,i]
	return b

	"""
	x = np.zeros(shape=b.shape)

	for i in range(L.shape[0]): 
		x[i] = b[i]
		for j in range(i):
			x[i] -= L[i,j]*x[j]
		x[i] = x[i]/L[i,i]

	return x
	"""

def column_oriented_forward_substitution(L,b):

	for j in range(L.shape[1]): 
		b[j] = b[j]/L[j,j]
		for i in range(j+1,L.shape[0]):
			b[i] -= L[i,j]*b[j]

	return b

def unit_row_oriented_forward_substitution(L,b):

	for i in range(L.shape[0]): 
		for j in range(i-1):
			b[i] -= L[i,j]*b[j]
	return b


def unit_column_oriented_forward_substitution(L,b):

	for j in range(L.shape[1]): 
		for i in range(j,L.shape[0]):
			b[i] -= L[i,j]*b[j]

	return b

#
# Ux = y
#

def row_oriented_backward_substitution(U,y):

	for i in range(U.shape[0]-1,-1,-1): 
		for j in range(i+1, U.shape[1]):
			y[i] -= U[i,j]*y[j]
		y[i] = y[i]/U[i,i]
		
	return y

	"""
	x = np.zeros(shape=y.shape)

	for i in range(U.shape[0]-1,-1,-1): 
		x[i] = y[i]
		for j in range(i+1, U.shape[1]):
			x[i] -= U[i,j]*x[j]

		x[i] = x[i]/U[i,i]
		
	return x
	"""

def column_oriented_backward_substitution(U,y):

	for j in range(U.shape[1]-1,-1,-1):
		y[j] = y[j]/U[j,j]
		for i in range(j):
			y[i] -= U[i,j]*y[j]
		
		
	return y

#
# Testing
#

if __name__ == "__main__":

	# create random matrix
	n = 5
	MAX_VAL = 100
	R = np.random.rand(n,n)*np.random.randint(low=1, high=MAX_VAL) # random matrix

	# Get upper part for upper and lower triangular matrices
	U = np.triu(R, 0) 
	L = np.tril(R, 0)

	# create random x
	x_true = np.random.rand(n)*np.random.randint(low=1, high=MAX_VAL)

	# find b and y 
	b1 = L @ x_true
	b2 = L @ x_true
	y1 = U @ x_true
	y2 = U @ x_true

	# test substitution methods
	x_rf = row_oriented_forward_substitution(L,b1)
	x_cf = column_oriented_forward_substitution(L,b2)
	x_rb = row_oriented_backward_substitution(U,y1)
	x_cb = column_oriented_backward_substitution(U,y2)

	# display average errors
	print(np.average(abs(x_true - x_rf)))
	print(np.average(abs(x_true - x_cf)))
	print(np.average(abs(x_true - x_rb)))
	print(np.average(abs(x_true - x_cb)))

	# testing unit forward substitution
	L = np.tril(R, 0)
	L_unit = np.copy(L)
	np.fill_diagonal(L_unit, 1)

	b3 = L_unit @ x_true
	b4 = L_unit @ x_true

	x_urf = unit_row_oriented_forward_substitution(L, b3) # asumes unit lower triangular
	x_ucf = unit_column_oriented_forward_substitution(L, b4) # asumes unit lower triangular
	x_ul_true = row_oriented_forward_substitution(L_unit, b3) # passed actual lower priangular

	print(np.average(abs(x_urf - x_ul_true)))
	print(np.average(abs(x_urf - x_ul_true)))
