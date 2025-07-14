import numpy as np
import matplotlib.pyplot as plt

def bs_characteristic_function(u, S0, r, sigma, T):
    return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma ** 2) * T)
                  - 0.5 * sigma ** 2 * u ** 2 * T)

def carr_madan_fft(S0, r, T, alpha, N=4096, B=1000, sigma=0.2):
    eta = B / N  # spacing in frequency domain
    lambd = 2 * np.pi / (N * eta)  # spacing in log-strike domain
    beta = np.log(S0) - N * lambd / 2  # center of strike grid

    # v: frequency domain grid
    v = np.arange(N) * eta

    # Characteristic function
    cf = bs_characteristic_function(v - (alpha + 1) * 1j, S0, r, sigma, T)

    # Carr-Madan integrand
    numerator = np.exp(-r * T) * cf
    denominator = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    integrand = numerator / denominator * np.exp(1j * v * beta) * eta

    # Simpson's rule weights for accuracy
    simpson_weights = (3 + (-1)**np.arange(N)) / 3
    simpson_weights[0] = 1
    integrand *= simpson_weights

    # FFT
    fft_values = np.fft.fft(integrand).real

    # Recover strikes and prices
    K = np.exp(beta + np.arange(N) * lambd)
    C = np.exp(-alpha * (np.log(K))) * fft_values / np.pi
    return K, C


K, C = carr_madan_fft(S0=100, r=0.05, T=1.0, alpha=1.5)

plt.plot(K, C)
plt.xlabel('Strike Price')
plt.ylabel('Call Price')
plt.title('European Call Prices via Carr-Madan FFT')
plt.grid()
plt.show()
