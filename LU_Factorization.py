#!/usr/bin/env python3

"""

	Gaussian Elimination
	LU Factorization

	# LU

	Ax = b   ==>    a.) Ux=y,  
					b.) Ly=b
"""

# my lib
from fwd_bwd_substitution import row_oriented_backward_substitution as bsub
from fwd_bwd_substitution import unit_row_oriented_forward_substitution as ufsub

# external
import numpy as np

def naive_gaussian_elimination(A,b):
	# overwrite A and b with U (upper triangular) and c | Ux = c:
	n = A.shape[0]
	for k in range(n-1):
		for i in range(k+1,n):
			m = A[i,k]/A[k,k]
			for j in range(k+1,n):
				A[i,j] -= m*A[k,j]
			b[i] -= m*b[k]
	return A,b

def naive_LU_factorization(A):
	# overwrite A with U (upper triangular) and (unit Lower triangular) L 
	n = A.shape[0]
	for k in range(n-1):
		for i in range(k+1,n):
			A[i,k] = A[i,k]/A[k,k]
			for j in range(k+1,n):
				A[i,j] -= A[i,k]*A[k,j]
	return A


#
# Testing
#

if __name__ == "__main__":

	# create random matrix
	n = 3
	MAX_VAL = 10

	# create random matrix A
	A1 = np.random.rand(n,n)*np.random.randint(low=1, high=MAX_VAL)
	A2 = np.copy(A1)

	# create random x
	x_true = np.random.rand(n)*np.random.randint(low=1, high=MAX_VAL)

	# find b
	b = A1 @ x_true

	# naive gaussian elimination and backward substitution
	U, c = naive_gaussian_elimination(A1,b)
	x = bsub(U,c)

	# display average error
	print(np.average(abs(x_true - x)))

	#
	# LU testing
	# 

	b = A2 @ x_true

	# gives predicted result
	A_LU = naive_LU_factorization(A2)

	# solve for y then x
	y = ufsub( A_LU, b ) # Ly=b; assume L is unit lower triangular 
	x = bsub(  A_LU, y ) # Ux=y; using normal back substitution
	
	# display average error
	print(np.average(abs(x_true - x)))

	# mpi cart 
	# hdf5
