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


if __name__ == '__main__':
	# the two sizes mentioned in the worksheet
	N = 1000
	M = [10, 100, 1000]
	
	dt = 1./N
	t 	= np.range(0, 1+dt, dt)

	# first, use the standard defition to generate a realization with N samples

	# generate random variables zeta

	# generate Wiener processes

	# plot processes over time

	# use the KL expansion to approximation the Wiener process
	
	# use the same random variables for all M
