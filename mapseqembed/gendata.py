import torch
import torch.distributions as tdist
from stepapprox import unitboxcar
import numpy

class Particle:
    def __init__(self,mu,rad):
        self.mu = mu
        self.rad = rad

class GenerateSynthData:
    def __init__(self,numASVs = 5,D = 3,avgNumReadsParticle = 50.0,numParticles=100):
        self.numASVs = numASVs
        self.D = D
        self.avgNumReadsParticle = avgNumReadsParticle
        self.numParticles = numParticles
        self.particles = []

        self.mu_rad = numpy.log(1.0)
        self.mu_std = 1.0
        # annealing parameter for unit step approximation
        self.step_approx = 32
        # compound Dirichlet concentration
        self.conc = 10.0

    def gen_data(self):
        # sample overall relative abundances of ASVs from a Dirichlet distribution
        self.ASV_rel_abundance = tdist.Dirichlet(torch.ones(self.numASVs)).sample()

        # sample spatial embedding of ASVs
        self.w = torch.zeros(self.numASVs,self.D)
        w_prior = tdist.MultivariateNormal(torch.zeros(self.D), torch.eye(self.D))

        for o in range(0,self.numASVs):
            self.w[o,:] = w_prior.sample()

        self.data = torch.zeros(self.numParticles,self.numASVs)

        num_nonempty = 0

        mu_prior = tdist.MultivariateNormal(torch.zeros(self.D), torch.eye(self.D))
        rad_prior = tdist.LogNormal(torch.tensor([self.mu_rad]),torch.tensor([self.mu_std]))

        # replace with neg bin prior
        num_reads_prior = tdist.Poisson(torch.tensor([self.avgNumReadsParticle]))

        while (num_nonempty < self.numParticles):
            # sample center
            mu = mu_prior.sample()
            rad = rad_prior.sample()

            zr = torch.zeros(1,self.numASVs,dtype=torch.float64)
            for o in range(0,self.numASVs):
                p = mu-self.w[o,:]
                p = torch.pow(p,2.0)/rad
                p = (torch.sum(p)).sqrt()
                zr[0,o] = unitboxcar(p, 0.0, 2.0, self.step_approx)

            if torch.sum(zr) > 0.95:
                particle = Particle(mu,self)
                particle.zr = zr
                self.particles.append(particle)

                # renormalize particle abundances
                rn = self.ASV_rel_abundance*zr
                rn = rn/torch.sum(rn)

                # sample relative abundances for particle
                part_rel_abundance = tdist.Dirichlet(rn*self.conc).sample()

                # sample number of reads for particle
                # (replace w/ neg bin instead of Poisson)
                num_reads = num_reads_prior.sample().long().item()
                particle.total_reads = num_reads

                particle.reads = tdist.Multinomial(num_reads,probs=part_rel_abundance).sample()

                num_nonempty += 1
