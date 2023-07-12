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
    W = np.zeros(len(t))
    for i in range(len(t)):
        approx = 0.
        for n in range(1, M):
            approx += np.sin((n+0.5)*np.pi*t[i]) * zeta[n] / ((n+0.5)*np.pi)
        W[i] = np.sqrt(2) * approx
    return W

#second way of defining approximation (different zeta)
def KL_expansion(t, eigvals):
    n = len(eigvals)
    X = np.random.normal(0, 1, n)  # Generate standard Gaussian random variables
    psi_t = np.zeros_like(t)  # Initialize KL expansion

    for i in range(n):
        psi_t +=  np.sqrt(eigvals[i]) * np.sin(np.pi * (i+1) * t)
    #psi_t += np.sqrt(2) * np.sqrt(eigvals[i]) * np.sin(np.pi * (i - 0.5) * t) * X[i]
    return psi_t

def plot_w(t, W, M):
    plot(t, W)
    xlabel("Time")
    ylabel("Wiener Process")
    name =  "Wiener Process over time, M = " + str(M)
    title(name)
    show()

def plot_kl(t, W, M):
    plot(t, W)
    xlabel("Time")
    ylabel("Wiener Process KL Approximation")
    name = "Wiener Process  KL Approximation over time, M = " + str(M)
    title(name)
    show()


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
    plot_w(t, W, N)


    # use the KL expansion to approximation the Wiener process
    zeta2 = np.random.normal(0, 1, N+1)
    W_KL_approx = np.zeros(len(t))
    W_KL_approx = WP_KL_approx(zeta2,t, N)
    plot_kl(t, W_KL_approx, N)


    #eigenvalues
    eigenvalues = np.zeros(N)
    for i in range(N):
        eigenvalues[i] = 1/(((i-0.5) **2) * np.pi **2)

    xlabel('Number in array')
    ylabel('Eigenvalue')
    title('Eigenvalues')
    plot(eigenvalues)
    show()


    """W_appr_KL = np.zeros(len(t))
    for i in range(len(t)):
        W_appr_KL[i] = KL_expansion(t[i], eigenvalues)
    plot(t, W_appr_KL)
    xlabel("Time")
    ylabel("Wiener Process KL Approximation")
    title("Wiener Process KL Expansion Approximation over time")
    show()
    """

    # use the same random variables for all M
    for m in M:
        W_KL_approx = WP_KL_approx(zeta2, t, m)
        plot_kl(t, W_KL_approx, m)



