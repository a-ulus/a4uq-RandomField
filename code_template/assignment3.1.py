import numpy as np
from matplotlib.pyplot import *

# standard definition of the Wiener process
# W_0 = 0, W_{t + dt} = W_t + zeta_t. zeta_t ~ N(0, dt)
def WP_std_def(zeta, t):
	W = np.zeros(len(t))
	for i in range(1, len(t)):
		W[i] = W[i-1] + zeta[i-1] * np.sqrt(t[i] - t[i-1])
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

	# first, use the standard defition to generate a realization with N samples

	# generate random variables zeta
	zeta = np.random.normal(0, np.sqrt(dt), N)
	# generate Wiener processes
	W = WP_std_def(zeta, t)
	# plot processes over time
	#for i in range(N-1):
		#plot(t[i], W[i], label=f"M = {M[i]}")
	plot(t, W)
	xlabel("Time")
	ylabel("Wiener Process")
	legend()
	show()

	# use the KL expansion to approximation the Wiener process
	
	# use the same random variables for all M
