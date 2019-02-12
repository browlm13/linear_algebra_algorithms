#!/usr/bin/env python3

"""

	Gaussian Elimination

"""

# my lib
from forward_backward_substitution import row_oriented_backward_substitution as bsub

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


#
# Testing
#

if __name__ == "__main__":

	# create random matrix
	n = 5
	MAX_VAL = 100

	# create random matrix A
	A = np.random.rand(n,n)*np.random.randint(low=1, high=MAX_VAL)

	# create random x
	x_true = np.random.rand(n)*np.random.randint(low=1, high=MAX_VAL)

	# find b
	b = A @ x_true

	# naive gaussian elimination and backward substitution
	U, c = naive_gaussian_elimination(A,b)
	x = bsub(U,c)

	# display average error
	print(np.average(abs(x_true - x)))
