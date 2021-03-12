import torch
import pyro
import pyro.distributions as dist
import math

## Let's start with just the generative model for bulk 16S data, to get the ball rolling

class SpatialEmbed:
    def __init__(self, numOTUs, numReads, D = 3):
        self.D = D
        self.numOTUs = numOTUs
        self.numReads = numReads

        self.numComponents_bulkPrior = 10
        # in real model, we'll estimate some hyperparameters from the data
        self.DMD_scale = 100.0
        self.mu = torch.linspace(0,math.log10(self.DMD_scale),steps=self.numComponents_bulkPrior)
        self.psi = math.log(10.0)
        self.bulkDirichletParam = torch.ones(self.numComponents_bulkPrior)/self.numComponents_bulkPrior

    def model(self):
        pi = pyro.sample("pi", dist.Dirichlet(self.bulkDirichletParam))

        c = []
        rho = torch.zeros(self.numOTUs)
        for i in pyro.plate("OTUs_rho_plate", self.numOTUs):
            c.append(pyro.sample("c_{}".format(i), dist.Categorical(pi)))
            rho[i] = torch.exp(pyro.sample("rho_prime".format(i), dist.Normal(self.mu[c[i]], self.psi)))

        # sample bulk 16S reads
        # would be conditioned on data unless fully generating
        q = pyro.sample("q", dist.DirichletMultinomial(rho,total_count=self.numReads))

        return pi,c,rho,q

model = SpatialEmbed(10,1000,3)
pi,c,rho,q = model.model()
print(rho)
print(q)
