import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
import gendata as gd
import numpy
from tqdm import tqdm

#pyro.enable_validation(True)

# analytical approximation to the Heaviside with a logistic function
def heavisideLogistic(x,approx_param):
    y = 1.0/(1.0+torch.exp(-2.0*approx_param*x))
    return y

def unitboxcar(x,mu,l,approx_param):
    # parameterize boxcar function by the center and length
    y = heavisideLogistic(x-mu+l/2.0,approx_param) - heavisideLogistic(x-mu-l/2.0,approx_param)
    return y

class SpatialEmbed:
    def __init__(self, data, D = 3):
       self.numASVs = data.numASVs
       self.D = D
       self.numParticles = len(data.particles)
       self.particles = data.particles

       # hyperparameters for particle radii
       self.mean_rad = numpy.log(1.0)
       self.std_rad = 1.0

       # annealing parameter for unit step approximation
       self.step_approx = 32

       # hyperparameters for read DMD concentration parameter prior (lognormal)
       self.mean_DMD_conc = numpy.log(5.0)
       self.std_DMD_conc = numpy.log(25.0)

    def calc_zr(self, w, mu, rad):
        p = mu-w
        p = torch.pow(p,2.0)/rad
        #p = (torch.sum(p,dim=1)).sqrt()
        p = (torch.sum(p)).sqrt()
        return unitboxcar(p, 0.0, 2.0, self.step_approx)

    def model(self):
        # sample overall ASV abundances
        ASV_rel_abundance = pyro.sample("ASV_rel_abundance", dist.Dirichlet(torch.ones(self.numASVs)))

        # sample DMD concentration
        DMD_conc = pyro.sample("DMD_conc", dist.LogNormal(torch.tensor([self.mean_DMD_conc]),torch.tensor([self.std_DMD_conc])))

        # sample ASV embeddings
        #with pyro.plate("ASV_embedding", self.numASVs):
        #    w = pyro.sample("w", dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)))

        w = []
        for i in pyro.plate("ASV_embedding", self.numASVs):
            w.append(pyro.sample("w_{}".format(i), dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D))))

        # sample particle centers and radii
        #with pyro.plate("particle_centers", self.numParticles):
        #    mu = pyro.sample("mu", dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)))
        #    rad = pyro.sample("rad", dist.LogNormal(torch.tensor([self.mean_rad]),torch.tensor([self.std_rad])))

        mu = []
        rad = []
        for i in pyro.plate("particle_centers", self.numParticles):
            mu.append(pyro.sample("mu_{}".format(i), dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D))))
            rad.append(pyro.sample("rad_{}".format(i), dist.LogNormal(torch.tensor([self.mean_rad]),torch.tensor([self.std_rad]))))

        particle_reads = []

        # sample data (sequencing reads)
        for i in pyro.plate('data',self.numParticles):
            zr = torch.zeros(len(w))
            for j in range(0,len(w)):
                zr[j] = self.calc_zr(w[j],mu[i],rad[i])

            #zr = self.calc_zr(w,mu[i,:],rad[i])

            # renormalize particle abundances
            rn = ASV_rel_abundance*zr
            rn = rn/torch.sum(rn)

            # sample reads from Multinomial
            particle_reads.append(pyro.sample("particle_reads_{}".format(i), dist.DirichletMultinomial(rn*DMD_conc,total_count=self.particles[i].total_reads), obs=self.particles[i].reads))

        return ASV_rel_abundance, DMD_conc, w, mu, rad, particle_reads

    def guide(self):
        # parameters for ASV_rel_abundance VI distribution
        VI_Dirichlet_ASV_rel_abundance = pyro.param("VI_Dirichlet_ASV_rel_abundance", lambda: dist.Dirichlet(torch.ones(self.numASVs)).sample(), constraint=dist.constraints.positive)

        # parameters for DMD_conc VI distribution
        ## this is producing very large values, maybe not a good distribution
        VI_mean_DMD_conc = pyro.param("VI_mean_DMD_conc", lambda: dist.LogNormal(torch.tensor([self.mean_DMD_conc]),torch.tensor([self.std_DMD_conc])).sample(), constraint=dist.constraints.positive)
        VI_std_DMD_conc = pyro.param("VI_std_DMD_conc", lambda: dist.Uniform(torch.tensor([self.std_DMD_conc/10.0]),torch.tensor([self.std_DMD_conc*10.0])).sample(), constraint=dist.constraints.positive)

        # parameters for ASV embeddings
        #VI_mean_w = pyro.param("VI_mean_w", lambda: dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)).sample([self.numASVs]))
        #VI_var_w = pyro.param("VI_var_w",torch.ones([self.numASVs,self.D]),constraint=dist.constraints.positive)

        # parameters for particle centers
        #VI_mean_mu = pyro.param("VI_mean_mu", lambda: dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)).sample([self.numParticles]))
        #VI_var_mu = pyro.param("VI_var_mu",torch.ones([self.numParticles,self.D]),constraint=dist.constraints.positive)

        # parameters for radii
        #VI_mean_rad = pyro.param("VI_mean_rad", lambda: dist.LogNormal(torch.tensor([self.mean_rad]),torch.tensor([self.std_rad])).sample([self.numParticles]), constraint=dist.constraints.positive)
        #VI_std_rad = pyro.param("VI_std_rad", lambda: dist.Uniform(torch.tensor([self.std_rad/2.0]),torch.tensor([self.std_rad*2.0])).sample([self.numParticles]), constraint=dist.constraints.positive)

        # sample overall ASV abundances
        q_ASV_rel_abundance = pyro.sample("ASV_rel_abundance", dist.Dirichlet(VI_Dirichlet_ASV_rel_abundance))

        # sample DMD concentration
        q_DMD_conc = pyro.sample("DMD_conc", dist.LogNormal(VI_mean_DMD_conc, VI_std_DMD_conc))
        print(q_DMD_conc)
        print(VI_mean_DMD_conc)
        print(VI_std_DMD_conc)

        # sample ASV embeddings
        #with pyro.plate("ASV_embedding", self.numASVs):
        #    q_w = pyro.sample("w", dist.MultivariateNormal(VI_mean_w,torch.diag_embed(VI_var_w, offset=0, dim1=-2, dim2=-1)))

        for i in pyro.plate("ASV_embedding", self.numASVs):
            VI_mean_w = pyro.param("VI_mean_w_{}".format(i), lambda: dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)).sample())
            VI_var_w = pyro.param("VI_var_w_{}".format(i),torch.ones(self.D),constraint=dist.constraints.positive)

            q_w = pyro.sample("w_{}".format(i), dist.MultivariateNormal(VI_mean_w,torch.diag_embed(VI_var_w)))

        # sample particle centers and radii
        #with pyro.plate("particle_centers", self.numParticles):
        #    q_mu = pyro.sample("mu", dist.MultivariateNormal(VI_mean_mu,torch.diag_embed(VI_var_mu, offset=0, dim1=-2, dim2=-1)))
        #    q_rad = pyro.sample("rad", dist.LogNormal(torch.squeeze(VI_mean_rad), torch.squeeze(VI_std_rad)))

        for i in pyro.plate("particle_centers", self.numParticles):
            # parameters for particle centers
            VI_mean_mu = pyro.param("VI_mean_mu_{}".format(i), lambda: dist.MultivariateNormal(torch.zeros(self.D),torch.eye(self.D)).sample())
            VI_var_mu = pyro.param("VI_var_mu_{}".format(i),torch.ones(self.D),constraint=dist.constraints.positive)

            q_mu = pyro.sample("mu_{}".format(i), dist.MultivariateNormal(VI_mean_mu,torch.diag_embed(VI_var_mu)))

            # parameters for radii
            VI_mean_rad = pyro.param("VI_mean_rad_{}".format(i), lambda: dist.LogNormal(torch.tensor([self.mean_rad]),torch.tensor([self.std_rad])).sample(), constraint=dist.constraints.positive)
            VI_std_rad = pyro.param("VI_std_rad_{}".format(i), lambda: dist.Uniform(torch.tensor([self.std_rad/2.0]),torch.tensor([self.std_rad*2.0])).sample(), constraint=dist.constraints.positive)

            q_rad = pyro.sample("rad_{}".format(i), dist.LogNormal(VI_mean_rad,VI_std_rad))

    def train(self,num_iters):
        optim = Adam({"lr": 0.0001})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        losses = []

        pyro.clear_param_store()
        for j in tqdm(range(num_iters)):
            loss = svi.step()
            losses.append(loss)

data = gd.GenerateSynthData()
data.gen_data()
model = SpatialEmbed(data,3)
ASV_rel_abundance, DMD_conc, w, mu, rad, particle_reads = model.model()

model.train(10)
