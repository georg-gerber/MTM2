import numpy as np
import scipy.stats
import math
import scipy.special as sc
import matplotlib.pyplot as plt

from numpy.random import default_rng

rng = default_rng()

class UnivariateRule:
    def __init__(self,num_intervals):
        self.num_intervals = num_intervals
        self.breaks = np.zeros(self.num_intervals+1)
        self.v = np.zeros(self.num_intervals)
        self.alpha = np.zeros(self.num_intervals-1)

        # priors
        self.break_conc = 50
        self.sigma_v0 = 1.0
        self.sigma_alpha0 = 0.5
        self.sigma_alpha = 0.5

    def sample_from_prior(self):
        # sample breaks from a Dirichlet
        break_probs = rng.dirichlet(self.break_conc*np.ones(self.num_intervals)/(self.num_intervals))
        bt = np.cumsum(break_probs)
        self.breaks[1:(self.num_intervals+1)] = bt
        self.mid_points = (self.breaks[1:(self.num_intervals+1)] + self.breaks[0:self.num_intervals])/2.0
        #self.delta = np.diff(np.log(self.mid_points))
        self.delta = np.diff(self.mid_points)

        # sample initial value
        self.v[0] = np.exp(rng.normal(np.log(self.mid_points[0]),self.sigma_v0))
        self.alpha[0] = rng.normal(0,self.sigma_alpha0)
        self.alpha[1:(self.num_intervals-1)] = rng.normal(np.zeros(self.num_intervals-2),np.ones(self.num_intervals-2)*self.sigma_alpha*np.sqrt(self.delta[1:len(self.delta)]))
        self.alpha = np.cumsum(self.alpha)
        self.v[1:(self.num_intervals)] = np.exp(self.alpha)*self.delta
        self.v = np.cumsum(self.v)
        #self.v = np.exp(self.v)
        #self.v = np.log(1.0+np.exp(self.v))

    def plot(self):
        fig, ax = plt.subplots()
        for i in range(0,self.num_intervals):
            ax.plot([self.breaks[i], self.breaks[i+1]],[self.v[i], self.v[i]])
        ax.plot([0,1],[0,1])
        plt.show()

rule = UnivariateRule(5)
rule.sample_from_prior()
print(rule.breaks)
print(rule.v)
rule.plot()
