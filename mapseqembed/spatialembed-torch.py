import torch
import torch.distributions
import math

# analytical approximation to the Heaviside with a logistic function
def heavisideLogistic(x,approx_param):
    y = 1.0/(1.0+torch.exp(-2.0*approx_param*x))
    return y

def unitboxcar(x,mu,l,approx_param):
    # parameterize boxcar function by the center and length
    y = heavisideLogistic(x-mu+l/2.0,approx_param) - heavisideLogistic(x-mu-l/2.0,approx_param)
    return y

def negBinomial(mean,var):
    ## return NegativeBinomial parameterized by mean and variables
    return torch.distributions.NegativeBinomial(math.pow(mean,2.0)/(var-mean),1-(mean/var))

class SpatialEmbed:
    def __init__(self, numOTUs, numParticles, D = 3):
        self.D = D
        self.numOTUs = numOTUs
        self.numParticles = numParticles

        # annealing parameter for unit step approximation
        self.step_approx = 32.0

        self.numComponents_bulkPrior = 10
        # in real model, we'll estimate some hyperparameters from the data
        self.DMD_scale = 100.0
        self.mu = torch.linspace(0,math.log10(self.DMD_scale),steps=self.numComponents_bulkPrior)
        print(self.mu)
        self.psi = math.log(10.0)
        self.bulkDirichletParam = torch.ones(self.numComponents_bulkPrior)/self.numComponents_bulkPrior
        self.nu_w = 1.0
        self.nu_u = 1.0
        self.eta_lambda1 = 100.0
        self.eta_lambda2 = 100.0
        self.sigma_theta = 0.1
        self.sigma_psi = 0.1
        self.kappa = 0.1
        self.tau_q = 10000
        self.epsilon_q = 1000*1000
        self.tau_r = 100
        self.epsilon_r = 20*20

        # random variables
        self.pi = torch.zeros(self.numComponents_bulkPrior)
        self.c = torch.zeros(self.numOTUs,dtype=torch.long)
        self.rho = torch.zeros(self.numOTUs)
        self.w = torch.zeros(self.numOTUs,self.D)
        self.u = torch.zeros(self.numParticles,self.D)
        self.mlambda = torch.zeros(self.numParticles)
        self.theta_prime = torch.zeros(self.numParticles,self.numOTUs)
        self.q_hat = 0
        self.r_hat = torch.zeros(self.numParticles)

        # intermediate computations on random variables
        self.z = torch.zeros(self.numParticles,self.numOTUs)
        self.theta = torch.zeros(self.numParticles,self.numOTUs)

        # MCMC tuning hyperparameters
        self.w_std_tune = 0.5

        # MCMC iterations
        self.sample_prior_iters = 1000
        self.sample_prior_w_accept = 0

    def calc_z(self, w, mu, rad):
        p = mu-w
        p = torch.pow(p,2.0)
        p = ((torch.sum(p,dim=1)).sqrt())/rad
        return unitboxcar(p, 0.0, 2.0, self.step_approx)

    def generateSynthData(self):
        self.pi = torch.distributions.Dirichlet(self.bulkDirichletParam).sample()

        for i in range(0,self.numOTUs):
            self.c[i] = torch.distributions.Categorical(self.bulkDirichletParam).sample()
            self.rho[i] = torch.exp(torch.distributions.Normal(self.mu[self.c[i]], self.psi).sample())

        # sample bulk 16S reads
        qt = torch.distributions.Dirichlet(self.rho).sample()
        self.q_hat = int(negBinomial(self.tau_q, self.epsilon_q).sample().numpy().item())
        self.q = torch.distributions.Multinomial(self.q_hat,qt).sample()

        # initialize OTU centers
        w_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_w,2.0)*torch.eye(self.D))
        for i in range(0,self.numOTUs):
            self.w[i,:] = w_prior.sample()

        # initialize particles
        mlambda_prior = torch.distributions.Gamma(self.eta_lambda1,self.eta_lambda2)
        u_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_u,2.0)*torch.eye(self.D))
        for i in range(0,self.numParticles):
            self.mlambda[i] = mlambda_prior.sample()
            self.u[i,:] = u_prior.sample()
            self.r_hat[i] = int(negBinomial(self.tau_r, self.epsilon_r).sample().numpy().item())

        # compute z
        for i in range(0,self.numParticles):
            self.z[i,:] = self.calc_z(self.w, self.u[i,:], self.mlambda[i])

        # initialize theta
        for i in range(0,self.numParticles):
            self.theta_prime[i,:] = torch.distributions.Normal(torch.log(self.rho),torch.ones(self.numOTUs)*self.sigma_theta).sample()
            self.theta[i,:] = self.kappa * self.z[i,:] * torch.exp(self.theta_prime[i,:])

        print(torch.log(self.rho[0]))
        print(self.z[:,0]*100)

        #for i in range(0,self.sample_prior_iters):
        for i in range(0,100):
            j = 0
            #for j in range(0,self.numOTUs):
            print(i)
            self.sample_w_posterior(j)

        print(self.z[:,0]*100)

        #print(self.sample_prior_w_accept/(self.numOTUs*self.sample_prior_iters))
        print(self.sample_prior_w_accept/(self.sample_prior_iters))

    def sample_w_posterior(self,widx):
        ## update OTU centers using Metropolis-Hastings moves

        ## sample with Gaussian kernel
        w_new = torch.distributions.MultivariateNormal(self.w[widx,:], math.pow(self.w_std_tune,2.0)*torch.eye(self.D)).sample()
        # compute new z values
        z_new = self.calc_z(w_new, self.u, self.mlambda)
        theta_new = self.kappa * z_new * torch.exp(self.theta_prime[:,widx])

        # compute term that 'constrains' particle reads to aggregate to bulk reads
        stt_old = torch.sum(self.z[:,widx])
        st_old = torch.log(torch.sum(self.theta[:,widx])/(self.kappa*stt_old)+1.0)
        if stt_old == 0:
            st_old = torch.tensor(0.0)
        #print(st_old)

        stt_new = torch.sum(z_new)
        st_new = torch.log(torch.sum(theta_new)/(self.kappa*stt_new)+1.0)
        if stt_new == 0:
            st_new = torch.tensor(0.0)

        #ss_old = torch.sum(self.theta,dim=1)
        #ss_new = ss_old - self.theta[:,widx] + theta_new

        # finish these eventually to add reads
        #log_DMD_old = lgamma(ss_old) - lgamma(r_hat + ss_old)
        #log_DMD_new = lgamma(ss_new) - lgamma(r_hat + ss_new)

        dl = torch.distributions.Normal(torch.log(self.rho[widx]),self.sigma_psi)
        w_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_w,2.0)*torch.eye(self.D))

        ll_old = dl.log_prob(st_old) + w_prior.log_prob(self.w[widx,:])
        ll_new = dl.log_prob(st_new) + w_prior.log_prob(w_new)

        p_accept = torch.min(torch.cat((torch.exp(ll_new - ll_old).reshape(1),torch.tensor([1.0]))))
        move_sample = torch.rand(1)

        ## accept the move
        if move_sample.item() < p_accept.item():
            self.w[widx,:] = w_new
            self.z[:,widx] = z_new
            self.theta[:,widx] = theta_new
            self.sample_prior_w_accept += 1
            print(stt_old)
            print(stt_new)

model = SpatialEmbed(10,50,3)
model.generateSynthData()
#print(model.z[0,:]*100)
