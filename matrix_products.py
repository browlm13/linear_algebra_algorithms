
"""

	Matrix Product

	-Row-oriented matrix-vector product
	-Column-oriented matrix vector product
	-Matrix-Matrix Product

"""

# external
import numpy as np

def row_oriented_matrix_vector_product(A,x):

	b = np.zeros(shape=(A.shape[0],))
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			b[i] = b[i] + A[i,j]*x[j]

	return b


def column_oriented_matrix_vector_product(A,x):
	b = np.zeros(shape=(A.shape[0],))
	for j in range(A.shape[1]):
		for i in range(A.shape[0]):
			b[i] = b[i] + A[i,j]*x[j]

	return b

def matrix_matrix_product(A,X):

	B = np.zeros(shape=(A.shape[0],X.shape[1]))

	for i in range(A.shape[0]):
		for j in range(X.shape[1]):
			for k in range(A.shape[1]): 
				B[i,j] = B[i,j] + A[i,k]*X[k,j]

	return B


#
# Testing
#

if __name__ == "__main__":

	A = np.array([[3,2,1],[6,5,4],[9,8,7]])
	x = np.array([8,4,2])


	b = row_oriented_matrix_vector_product(A,x)
	b_true = A @ x

	print(b_true - b)


	b = column_oriented_matrix_vector_product(A,x)
	b_true = A @ x

	print(b_true - b)

	X = np.array([[8,4],[4,2],[2,1]])
	B = matrix_matrix_product(A,X)
	B_true = A @ X

	print(B-B_true)

	

