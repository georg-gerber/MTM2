import torch
import torch.distributions
import math

# continous approximation to max
def contMax(x,approx_param):
    xt = torch.exp(x*approx_param)
    if len(list(x.size())) == 1:
        return torch.sum(x*xt)/torch.sum(xt)
    return torch.sum(x*xt,dim=1)/torch.sum(xt,dim=1)

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

        # parameter for continuous max approximation
        self.contmax_approx = 10.0

        # small smoothing parameter for theta
        self.smooth_theta = 0.000001

        # in real model, we'll estimate some hyperparameters from the data
        self.eta_q1 = 100.0
        self.eta_q2 = 1.0
        self.tau_q = 10000
        self.epsilon_q = 1000*1000
        self.nu_w = 1.0
        self.nu_u = 1.0
        self.eta_lambda1 = 100.0
        self.eta_lambda2 = 100.0
        self.tau_r = 100
        self.epsilon_r = 20*20
        self.min_one_per_particle_prob_param = 100.0
        self.beta = 10.0

        # random variables
        self.kappa_q = torch.tensor([100.0])
        self.pi_prime = torch.zeros(self.numOTUs)
        self.q = torch.zeros(self.numOTUs)
        self.w = torch.zeros(self.numOTUs,self.D)
        self.u = torch.zeros(self.numParticles,self.D)
        self.mlambda = torch.zeros(self.numParticles)
        self.r_hat = torch.zeros(self.numParticles)
        self.theta_prime = torch.zeros(self.numParticles,self.numOTUs)

        # intermediate computations on random variables
        self.pi = torch.zeros(self.numOTUs)
        self.z = torch.zeros(self.numParticles,self.numOTUs)
        self.theta = torch.zeros(self.numParticles,self.numOTUs)
        self.particle_aggregation = torch.zeros(self.numOTUs)
        self.one_per_particle_prob = torch.zeros(self.numParticles)

        # MCMC tuning hyperparameters
        self.w_std_tune = 0.025
        self.u_std_tune = 0.1
        self.lambda_std_tune = 0.1

        # MCMC iterations
        self.sample_prior_iters = 500
        self.sample_w_accept = 0
        self.sample_u_accept = 0
        self.sample_lambda_accept = 0

    def calc_z(self, w, mu, rad):
        p = mu-w
        p = torch.pow(p,2.0)
        p = ((torch.sum(p,dim=1)).sqrt())/rad
        return unitboxcar(p, 0.0, 2.0, self.step_approx)

    def min_one_per_particle_prob(self,z):
        p = contMax(z,self.contmax_approx) - 0.95
        return 1/(1+torch.exp(-self.min_one_per_particle_prob_param*p))

    def generateSynthData(self):
        # generate bulk reads
        self.pi_prime = torch.distributions.Normal(torch.zeros(self.numOTUs),torch.ones(self.numOTUs)).sample()
        self.pi = torch.exp(self.pi_prime)
        self.pi = self.pi/torch.sum(self.pi)
        self.kappa_q = torch.distributions.Gamma(self.eta_q1, self.eta_q2).sample()
        self.q_hat = int(negBinomial(self.tau_q, self.epsilon_q).sample().numpy().item())
        self.q = torch.distributions.Multinomial(self.q_hat,self.kappa_q*self.pi).sample()

        # initialize OTU centers
        w_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_w,2.0)*torch.eye(self.D))
        for i in range(0,self.numOTUs):
            self.w[i,:] = w_prior.sample()

        # initialize particles
        mlambda_prior = torch.distributions.Gamma(self.eta_lambda1,self.eta_lambda2)
        u_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_u,2.0)*torch.eye(self.D))
        for i in range(0,self.numParticles):
            self.r_hat[i] = int(negBinomial(self.tau_r, self.epsilon_r).sample().numpy().item())
            mmp = 0.0
            while mmp <= 0.9:
                self.mlambda[i] = mlambda_prior.sample()
                self.u[i,:] = u_prior.sample()
                self.z[i,:] = self.calc_z(self.w, self.u[i,:], self.mlambda[i])
                mmp = torch.max(self.z[i,:])

        self.one_per_particle_prob = self.min_one_per_particle_prob(self.z)

        # initialize theta
        self.theta_prime = torch.distributions.Normal(torch.zeros(self.numParticles,self.numOTUs),torch.ones(self.numParticles,self.numOTUs)).sample()
        self.theta = torch.exp(self.theta_prime)*self.z + self.smooth_theta
        self.theta = (self.theta.t()/torch.sum(self.theta,dim=1)).t()

        self.particle_aggregation = self.aggregate_particle_abundances(self.theta)

        #attempts = 0
        #for i in range(0,100):
        #    for j in range(0,self.numOTUs):
        #        attempts += 1
        #        self.sample_w_posterior(j)

        #print(self.sample_prior_w_accept/attempts)

        print(self.pi)
        print(self.particle_aggregation)

        num_particle_attempts = 0
        num_OTU_attempts = 0
        for i in range(0,self.sample_prior_iters):

            for j in range(0,self.numOTUs):
                    num_OTU_attempts += 1
                    self.sample_w_posterior(j)

            for j in range(self.numParticles):
                num_particle_attempts += 1
                self.sample_u_posterior(j)
                self.sample_lambda_posterior(j)

        print(self.particle_aggregation)

        print(self.sample_w_accept/num_OTU_attempts)
        print(self.sample_u_accept/num_particle_attempts)
        print(self.sample_lambda_accept/num_particle_attempts)
        print(torch.sum(self.z,dim=1))
        print(self.mlambda)

    def aggregate_particle_abundances(self,mytheta):
        return torch.sum(mytheta,dim=0)/self.numParticles

    def sample_w_posterior(self,widx,data_present = False):
        ## update OTU center using Metropolis-Hastings moves

        ## sample with Gaussian kernel
        w_new = torch.distributions.MultivariateNormal(self.w[widx,:], math.pow(self.w_std_tune,2.0)*torch.eye(self.D)).sample()
        # compute new z values
        z_new = self.z.clone()
        z_new[:,widx] = self.calc_z(w_new, self.u, self.mlambda)
        theta_new = torch.exp(self.theta_prime)*z_new + self.smooth_theta
        theta_new = (theta_new.t()/torch.sum(theta_new,dim=1)).t()

        # compute aggregation across particles
        agg_new = torch.sum(theta_new,dim=0)/self.numParticles

        # compute 'constraint' on >= 1 OTU per particle
        one_per_particle_new = self.min_one_per_particle_prob(z_new)

        w_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_w,2.0)*torch.eye(self.D))
        ll_old = w_prior.log_prob(self.w[widx,:]) + torch.sum(torch.log(self.one_per_particle_prob)) + torch.sum((self.beta*self.q-1.0)*torch.log(self.particle_aggregation))
        ll_new = w_prior.log_prob(w_new) + torch.sum(torch.log(one_per_particle_new)) + torch.sum((self.beta*self.q-1.0)*torch.log(agg_new))

        p_accept = torch.min(torch.cat((torch.exp(ll_new - ll_old).reshape(1),torch.tensor([1.0]))))
        move_sample = torch.rand(1)

        ## accept the move
        if move_sample.item() < p_accept.item():
            self.w[widx,:] = w_new
            self.z = z_new
            self.theta = theta_new
            self.sample_w_accept += 1
            self.particle_aggregation = agg_new
            self.one_per_particle_prob = one_per_particle_new
            #print('accept')
            #print(p_accept.item())
            #print(self.pi)
            #print(agg_new)

    def sample_u_posterior(self,uidx,data_present = False):
        ## update particle center using Metropolis-Hastings moves

        ## sample with Gaussian kernel
        u_new = torch.distributions.MultivariateNormal(self.u[uidx,:], math.pow(self.u_std_tune,2.0)*torch.eye(self.D)).sample()

        z_new = self.calc_z(self.w, u_new, self.mlambda[uidx])
        theta_new = torch.exp(self.theta_prime[uidx,:])*z_new + self.smooth_theta
        theta_new = theta_new/torch.sum(theta_new)
        one_per_particle_new = self.min_one_per_particle_prob(z_new)
        agg_new = self.particle_aggregation - self.theta[uidx,:]/self.numParticles + theta_new/self.numParticles

        u_prior = torch.distributions.MultivariateNormal(torch.zeros(self.D), math.pow(self.nu_u,2.0)*torch.eye(self.D))
        ll_old = u_prior.log_prob(self.u[uidx,:]) + torch.log(self.one_per_particle_prob[uidx]) + torch.sum((self.beta*self.q-1.0)*torch.log(self.particle_aggregation))
        ll_new = u_prior.log_prob(u_new) + torch.log(one_per_particle_new) + torch.sum((self.beta*self.q-1.0)*torch.log(agg_new))

        p_accept = torch.min(torch.cat((torch.exp(ll_new - ll_old).reshape(1),torch.tensor([1.0]))))
        move_sample = torch.rand(1)

        ## accept the move
        if move_sample.item() < p_accept.item():
            self.u[uidx,:] = u_new
            self.z[uidx,:] = z_new
            self.theta[uidx,:] = theta_new
            self.sample_u_accept += 1
            self.particle_aggregation = agg_new
            self.one_per_particle_prob[uidx] = one_per_particle_new
            #print('accept')
            #print(p_accept.item())
            #print(self.pi)
            #print(agg_new)

    def sample_lambda_posterior(self,uidx,data_present = False):
        ## update particle radis using Metropolis-Hastings moves

        ## sample with Gaussian kernel
        lambda_new = torch.exp(torch.distributions.Normal(torch.log(self.mlambda[uidx]), self.lambda_std_tune).sample())

        z_new = self.calc_z(self.w, self.u[uidx,:], lambda_new)
        theta_new = torch.exp(self.theta_prime[uidx,:])*z_new + self.smooth_theta
        theta_new = theta_new/torch.sum(theta_new)
        one_per_particle_new = self.min_one_per_particle_prob(z_new)
        agg_new = self.particle_aggregation - self.theta[uidx,:]/self.numParticles + theta_new/self.numParticles

        lambda_prior = torch.distributions.Gamma(self.eta_lambda1,self.eta_lambda2)
        ll_old = lambda_prior.log_prob(self.mlambda[uidx]) + torch.log(self.one_per_particle_prob[uidx]) + torch.sum((self.beta*self.q-1.0)*torch.log(self.particle_aggregation))
        ll_new = lambda_prior.log_prob(lambda_new) + torch.log(one_per_particle_new) + torch.sum((self.beta*self.q-1.0)*torch.log(agg_new))

        p_accept = torch.min(torch.cat((torch.exp(ll_new - ll_old).reshape(1),torch.tensor([1.0]))))
        move_sample = torch.rand(1)

        ## accept the move
        if move_sample.item() < p_accept.item():
            self.mlambda[uidx] = lambda_new
            self.z[uidx,:] = z_new
            self.theta[uidx,:] = theta_new
            self.sample_lambda_accept += 1
            self.particle_aggregation = agg_new
            self.one_per_particle_prob[uidx] = one_per_particle_new
            #print('accept')
            #print(p_accept.item())
            #print(self.pi)
            #print(agg_new)

model = SpatialEmbed(10,50,3)
model.generateSynthData()
#print(model.q)
#print(model.theta)
