import numpy as np
import scipy.stats
import math
import scipy.special as sc
import matplotlib.pyplot as plt

from numpy.random import default_rng

rng = default_rng()

start_value = 0
last_value = 1
num_intervals = 20

# sample breaks from a Dirichlet
breaks_prop = rng.dirichlet(25*np.ones(num_intervals)/num_intervals)
bt = np.cumsum(breaks_prop*(last_value-start_value)) + start_value
breaks = bt[0:(num_intervals-1)]

# sample values on num_intervals
v = np.zeros(num_intervals)
alpha = np.zeros(num_intervals-1)
sigma_v = 0.1
#sigma_alpha0 = 0.25
sigma_alpha0 = 0.01
sigma_alpha = 0.01
mu_gamma = np.log(0.5)
sigma_gamma = 0.1
#v[0] = np.exp(rng.normal(0,sigma_v))
v[0] = rng.normal(0,sigma_v)
alpha[0] = rng.normal(0,sigma_alpha0)
alpha[1:(num_intervals-1)] = rng.normal(np.zeros(num_intervals-2),np.ones(num_intervals-2)*sigma_alpha)
alpha = np.cumsum(alpha)
gamma = np.exp(rng.normal(np.ones(num_intervals-1)*mu_gamma,np.ones(num_intervals-1)*sigma_gamma))
v[1:num_intervals] = alpha*gamma
v = np.exp(np.cumsum(v))
fig, ax = plt.subplots()
breaks_expand = np.zeros(num_intervals+1)
breaks_expand[1:num_intervals] = breaks
#breaks_expand[0] = np.max([start_value-1,0]);
breaks_expand[0] = start_value-0.1
breaks_expand[num_intervals] = last_value + 0.1
mid_points = (breaks_expand[1:(num_intervals+1)] + breaks_expand[0:num_intervals])/2.0
v = v * mid_points
print(breaks_expand)
print(v)
for i in range(0,num_intervals):
    ax.plot([breaks_expand[i], breaks_expand[i+1]],[v[i], v[i]])
ax.plot([breaks_expand[0], breaks_expand[num_intervals]],[breaks_expand[0], breaks_expand[num_intervals]])
#ax.set_ylim([-5,5])
plt.show()
