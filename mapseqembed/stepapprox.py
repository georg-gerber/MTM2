import numpy as np

# analytical approximation to the Heaviside with a logistic function
def heavisideLogistic(x,approx_param):
    y = 1/(1+np.exp(-2.0*approx_param*x))
    return y

def unitboxcar(x,mu,l,approx_param):
    # parameterize boxcar function by the center and length
    y = heavisideLogistic(x-mu+l/2.0,approx_param) - heavisideLogistic(x-mu-l/2.0,approx_param)
    return y
