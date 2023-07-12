import numpy as np
from matplotlib.pyplot import *

# standard definition of the Wiener process
# W_0 = 0, W_{t + dt} = W_t + zeta_t. zeta_t ~ N(0, dt)
def WP_std_def(zeta, t):
	W = None

	return W

# KL approximation of the Wiener process
def WP_KL_approx(zeta, t, M):
	W = None

	return W


def karhunen_loeve_eigenvalues(M):
	# Generate a random covariance matrix
	covariance_matrix = np.random.rand(M, M)
	covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2  # Make it symmetric

	# Compute the eigenvalues of the covariance matrix
	eigenvalues = np.linalg.eigvalsh(covariance_matrix)

	return eigenvalues


if __name__ == '__main__':
	# the two sizes mentioned in the worksheet
	N = 1000
	M = [10, 100, 1000]
	
	dt = 1./N
	t 	= np.arange(0, 1+dt, dt)

	# Parameters
	m = 1000

	# Compute eigenvalues
	eigenvalues = karhunen_loeve_eigenvalues(m)
	print(eigenvalues)
	eigenvalues_wo_outliers = [item for item in eigenvalues if item< 100]

	# Plot the eigenvalues
	plot(eigenvalues, 'bo')
	xlabel('Eigenvalue index')
	ylabel('Eigenvalue')
	title('Eigenvalues of Karhunen-Loeve Expansion')
	show()

	# Plot the eigenvalues without outliers
	plot(eigenvalues_wo_outliers, 'bo')
	xlabel('Eigenvalue index')
	ylabel('Eigenvalue')
	title('Eigenvalues without outliers of Karhunen-Loeve Expansion')
	show()

	# first, use the standard defition to generate a realization with N samples

	# generate random variables zeta

	# generate Wiener processes

	# plot processes over time

	# use the KL expansion to approximation the Wiener process
	
	# use the same random variables for all M
