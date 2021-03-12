import numpy as np
import scipy.stats
import math
import scipy.special as sc

from numpy.random import default_rng
#rng = default_rng(seed=1)
rng = default_rng()

# continous approximation to max
def contMax(x,approx_param):
    xt = np.exp(x*approx_param)
    if x.ndim == 1:
        return np.sum(x*xt)/np.sum(xt)
    return np.sum(x*xt,axis=1)/np.sum(xt,axis=1)

# analytical approximation to the Heaviside with a logistic function
def heavisideLogistic(x,approx_param):
    y = 1.0/(1.0+np.exp(-2.0*approx_param*x))
    return y

def unitboxcar(x,mu,l,approx_param):
    # parameterize boxcar function by the center and length
    y = heavisideLogistic(x-mu+l/2.0,approx_param) - heavisideLogistic(x-mu-l/2.0,approx_param)
    return y

def negBinomialSample(mean,var):
    ## return NegativeBinomial sample parameterized by mean and variance
    return rng.negative_binomial(mean**2/(var-mean),(mean/var))

def logNormalLike(x,mu,sigma):
    ## computes the log PDF of a normal distribution
    ## this function is used for sampling and does not return the constants
    ## return -np.log(sigma) -0.5*np.log(2*math.pi) - 0.5*np.power((x-mu)/sigma,2.0)
    return -0.5*np.power((x-mu)/sigma,2.0)

def logGammaLike(x,k,theta):
    ## computes log PDF of a gamma distribution
    ## note that parameterization in NumPy is
    ## p(x) = x^{k-1} e^{-x/theta}/(theta^k Gamma(k))
    ## this function is used for sampling and does not return the constants
    return (k-1)*np.log(x) -x/theta

def logDirichletLike(x,alpha):
    return (sc.gammaln(np.sum(alpha)) - np.sum(sc.gammaln(alpha)) +
        np.sum((alpha-1)*np.log(x)))

def logDirichletMultinomial(x,alpha):
    ## computes log PMF of logDirichletMultinomial
    ## note that constants are omitted
    if x.ndim == 1:
        n = np.sum(x)
        alpha_ss = np.sum(alpha)
    else:
        n = np.sum(x,axis=1)
        alpha_ss = np.sum(alpha,axis=1)

    lpmf = sc.gammaln(alpha_ss) - sc.gammaln(n+alpha_ss)

    if x.ndim == 1:
        lpmf += np.sum(sc.gammaln(x+alpha) - sc.gammaln(alpha))
    else:
        lpmf += np.sum(sc.gammaln(x+alpha) - sc.gammaln(alpha),axis=1)

    return lpmf

class SpatialEmbed:
    def __init__(self, numOTUs, numParticles, D = 3, bulk_reads = None, particle_reads = None):
        self.D = D
        self.numOTUs = numOTUs
        self.numParticles = numParticles

        # annealing parameter for unit step approximation
        self.step_approx = 32.0

        # parameter for continuous max approximation
        self.contmax_approx = 20.0

        # small smoothing parameter for theta
        self.smooth_theta = 0.000001

        # in real model, we'll estimate some hyperparameters from the data
        self.eta_q1 = 100.0
        self.eta_q2 = 1.0
        self.tau_q = 10000
        self.epsilon_q = 1000**2
        self.nu_w = 0.05
        self.nu_u = 0.3
        self.eta_lambda1 = 100.0
        self.eta_lambda2 = 0.01
        self.tau_r = 100
        self.epsilon_r = 20**2
        self.min_one_per_particle_prob_param = 100.0
        self.beta = 1000.0
        self.eta_r1 = 100.0
        self.eta_r2 = 1.0

        # random variables
        self.kappa_q = 100.0
        self.pi_prime = np.zeros(self.numOTUs)
        self.q = np.zeros(self.numOTUs)
        self.w = np.zeros((self.numOTUs,self.D))
        self.u = np.zeros((self.numParticles,self.D))
        self.mlambda = np.zeros(self.numParticles)
        self.r_hat = np.zeros(self.numParticles)
        self.r = np.zeros((self.numParticles,self.numOTUs))
        self.omega_prime = np.zeros(self.numOTUs)
        self.kappa_r = 100.0

        # intermediate computations on random variables
        self.pi = np.zeros(self.numOTUs)
        self.omega = np.zeros(self.numOTUs)
        self.z = np.zeros((self.numParticles,self.numOTUs))
        self.theta_prime = np.zeros((numParticles,self.numOTUs))
        self.theta = np.zeros((numParticles, self.numOTUs))
        self.particle_aggregation = np.zeros(self.numOTUs)
        self.one_per_particle_prob = np.zeros(self.numParticles)
        self.total_bulk_reads = 0
        self.total_particle_reads = 0

        # MCMC tuning hyperparameters
        self.w_std_tune = 0.5
        self.u_std_tune = 0.8
        self.lambda_std_tune = 0.2
        self.pi_tune = 2000
        self.omega_tune = 4000

        # MCMC iterations
        self.sample_prior_iters = 1000
        self.sample_pi_accept = 0
        self.sample_omega_accept = 0
        self.sample_w_accept = 0
        self.sample_u_accept = 0
        self.sample_lambda_accept = 0

        if bulk_reads is not None:
            self.q = bulk_reads
            self.r = particle_reads
            self.init_from_data()

    def init_from_data(self):
        ## estimate bulk abundances pi from data
        self.pi = self.q/np.sum(self.q)
        self.pi_prime = np.log(self.pi)
        self.q_hat = np.sum(self.q)
        self.total_bulk_reads = np.sum(self.q)

        ## estimate particle abundances from data
        for i in range(0,self.numParticles):
            self.r_hat[i] = np.sum(self.r[i,:])
        self.total_particle_reads = np.sum(self.r_hat)

        self.init_OTU_centers()
        self.init_particles()
        self.init_omega()
        self.particle_aggregation = self.aggregate_particle_abundances(self.theta,self.particle_vol(self.mlambda,self.D))

    def calc_z(self, w, mu, rad):
        p = mu-w
        p = p**2
        p = np.sqrt(np.sum(p,axis=1))/rad
        return unitboxcar(p, 0.0, 2.0, self.step_approx)

    def min_one_per_particle_prob(self,z):
        p = contMax(z,self.contmax_approx) - 0.95
        return 1/(1+np.exp(-self.min_one_per_particle_prob_param*p))

    def aggregate_particle_abundances(self,mytheta,myvol):
        return np.sum((mytheta.transpose()*myvol).transpose(),axis=0)/np.sum(myvol)

    def particle_vol(self,mylambda,myD):
        ## compute volume of D-sphere without constants
        return np.power(mylambda,myD)

    def init_OTU_centers(self):
        # initialize OTU centers
        self.w = rng.multivariate_normal(np.zeros(self.D), math.pow(self.nu_w,2.0)*np.eye(self.D),self.numOTUs)

    def init_particles(self):
        for i in range(0,self.numParticles):
            mmp = 0.0
            while mmp <= 0.95:
                self.mlambda[i] = rng.gamma(self.eta_lambda1,self.eta_lambda2)
                self.u[i,:] = rng.multivariate_normal(np.zeros(self.D), math.pow(self.nu_u,2.0)*np.eye(self.D))
                self.z[i,:] = self.calc_z(self.w, self.u[i,:], self.mlambda[i])
                mmp = contMax(self.z[i,:],self.contmax_approx)

        self.one_per_particle_prob = self.min_one_per_particle_prob(self.z)

    def init_omega(self):
        # sample omega
        self.omega_prime = rng.normal(np.zeros(self.numOTUs),np.ones(self.numOTUs))
        self.omega = np.exp(self.omega_prime)
        self.omega = self.omega/np.sum(self.omega)

        # initialize theta
        self.theta_prime = self.omega*self.z + self.smooth_theta
        self.theta = (self.theta_prime.transpose()/np.sum(self.theta_prime,axis=1)).transpose()

    def generateSynthData(self):
        # generate bulk reads
        self.pi_prime = rng.normal(np.zeros(self.numOTUs),np.ones(self.numOTUs))
        self.pi = np.exp(self.pi_prime)
        self.pi = self.pi/np.sum(self.pi)
        self.kappa_q = rng.gamma(self.eta_q1, self.eta_q2)
        self.q_hat = negBinomialSample(self.tau_q,self.epsilon_q)
        qdd = rng.dirichlet(self.kappa_q*self.pi)
        self.q = rng.multinomial(self.q_hat,qdd)

        # generate total reads per particle
        for i in range(0,self.numParticles):
            self.r_hat[i] = negBinomialSample(self.tau_r, self.epsilon_r)

        self.init_OTU_centers()
        self.init_particles()
        self.init_omega()

        self.total_bulk_reads = np.sum(self.q)
        self.total_particle_reads = np.sum(self.r_hat)

        self.particle_aggregation = self.aggregate_particle_abundances(self.theta,self.particle_vol(self.mlambda,self.D))
        print("initial lambda ",self.mlambda)
        print("initial pi ",self.pi)
        print("initial omega ",self.omega)
        #print("initial aggregate ",self.particle_aggregation)

        # sample from priors using MH moves
        for i in range(0,self.sample_prior_iters):
            self.sample_w_posteriors()
            self.sample_u_posteriors()
            self.sample_lambda_posteriors()

        # sample particle reads
        self.kappa_r = rng.gamma(self.eta_r1, self.eta_r2)
        for i in range(0,self.numParticles):
            rdd = rng.dirichlet(self.kappa_r*self.theta[i,:])
            self.r[i,:] = rng.multinomial(self.r_hat[i],rdd)

        print("final aggregate ",self.particle_aggregation)
        print("final lambda ",self.mlambda)
        #print("r ",self.r)

        print("accept w",self.sample_w_accept/(self.numOTUs*self.sample_prior_iters))
        print("accept u",self.sample_u_accept/(self.numParticles*self.sample_prior_iters))
        print("accept lambda",self.sample_lambda_accept/(self.numParticles*self.sample_prior_iters))

    def sample_pi_posteriors(self):
        ## update bulk OTU proportions using MH moves

        ## sample from Dirichlet proposal
        no_zero = False
        while (no_zero is False):
            proposal = rng.dirichlet(self.pi_tune*self.pi)
            no_zero = np.all(proposal)

        proposal_log = np.log(proposal)

        pi_prior_old_loglikes = np.sum(logNormalLike(self.pi_prime,0,1.0))
        pi_prior_new_loglikes = np.sum(logNormalLike(proposal_log,0,1.0))

        prob_old_from_new = logDirichletLike(self.pi,self.pi_tune*proposal)
        prob_new_from_old = logDirichletLike(proposal,self.pi_tune*self.pi)

        ll_old = (pi_prior_old_loglikes + prob_new_from_old +
            logDirichletLike(self.particle_aggregation,self.beta*self.pi) +
            logDirichletMultinomial(self.q,self.pi*self.kappa_q))

        ll_new = (pi_prior_new_loglikes + prob_old_from_new +
            logDirichletLike(self.particle_aggregation,self.beta*proposal) +
            logDirichletMultinomial(self.q,proposal*self.kappa_q))

        ## sample probability for acceptance
        log_uniform = np.log(rng.random())

        accept = (log_uniform < ll_new - ll_old)

        if accept:
            self.pi = proposal
            self.pi_prime = proposal_log
            self.sample_pi_accept += 1

    def sample_omega_posteriors(self):
        ## update particle OTU proportions using MH moves

        m_particle_vol = self.particle_vol(self.mlambda,self.D)

        ## sample from Dirichlet proposal
        no_zero = False
        while (no_zero is False):
            proposal = rng.dirichlet(self.omega_tune*self.omega)
            no_zero = np.all(proposal)

        proposal_log = np.log(proposal)

        pi_prior_old_loglikes = np.sum(logNormalLike(self.omega_prime,0,1.0))
        pi_prior_new_loglikes = np.sum(logNormalLike(proposal_log,0,1.0))

        prob_old_from_new = logDirichletLike(self.omega,self.omega_tune*proposal)
        prob_new_from_old = logDirichletLike(proposal,self.omega_tune*self.omega)

        # compute new theta values
        theta_prime_new = proposal*self.z + self.smooth_theta
        theta_new = (theta_prime_new.transpose()/np.sum(theta_prime_new,axis=1)).transpose()

        # compute aggregation across particles
        agg_new = self.aggregate_particle_abundances(theta_new,m_particle_vol)

        ll_old = (pi_prior_old_loglikes + prob_new_from_old +
            logDirichletLike(self.particle_aggregation,self.beta*self.pi) +
            np.sum(logDirichletMultinomial(self.r,self.theta*self.kappa_r)))

        ll_new = (pi_prior_new_loglikes + prob_old_from_new +
            logDirichletLike(agg_new,self.beta*self.pi) +
            np.sum(logDirichletMultinomial(self.r,theta_new*self.kappa_r)))

        ## sample probability for acceptance
        log_uniform = np.log(rng.random())

        accept = (log_uniform < ll_new - ll_old)

        if accept:
            self.omega = proposal
            self.omega_prime = proposal_log
            self.theta_prime = theta_prime_new
            self.theta = theta_new
            self.particle_aggregation = agg_new
            self.sample_omega_accept += 1

    def sample_w_posteriors(self,data_present = False):
        ## update OTU centers using Metropolis-Hastings moves

        ## sample from Gaussian kernel
        proposals = rng.multivariate_normal(np.zeros(self.D), math.pow(self.w_std_tune*self.nu_w,2.0)*np.eye(self.D),self.numOTUs)
        w_proposals = self.w + proposals
        # the priors are independent Gaussians, so we can vectorize this easily
        w_prior_old_loglikes = np.sum(logNormalLike(self.w,0,self.nu_w),axis=1)
        w_prior_new_loglikes = np.sum(logNormalLike(w_proposals,0,self.nu_w),axis=1)

        ## sample probabilities for acceptance
        log_uniforms = np.log(rng.random(self.numOTUs))

        m_particle_vol = self.particle_vol(self.mlambda,self.D)

        for widx, (w_new, log_uniform, w_prior_old_loglike, w_prior_new_loglike) in enumerate(zip(w_proposals,log_uniforms, w_prior_old_loglikes, w_prior_new_loglikes)):
            # compute new z values
            z_new = self.z.copy()
            z_new[:,widx] = self.calc_z(w_new, self.u, self.mlambda)

            # compute new theta values
            theta_prime_new = self.omega*z_new + self.smooth_theta
            theta_new = (theta_prime_new.transpose()/np.sum(theta_prime_new,axis=1)).transpose()

            # compute aggregation across particles
            agg_new = self.aggregate_particle_abundances(theta_new,m_particle_vol)

            # compute 'constraint' on >= 1 OTU per particle
            one_per_particle_new = self.min_one_per_particle_prob(z_new)

            ll_old = w_prior_old_loglike + np.sum(np.log(self.one_per_particle_prob)) + np.sum((self.beta*self.pi-1.0)*np.log(self.particle_aggregation))
            ll_new = w_prior_new_loglike + np.sum(np.log(one_per_particle_new)) + np.sum((self.beta*self.pi-1.0)*np.log(agg_new))

            if data_present is True:
                ## need to account for data in posterior under the Dirichlet Multinomial
                ll_old += np.sum(logDirichletMultinomial(self.r,self.theta*self.kappa_r))
                ll_new += np.sum(logDirichletMultinomial(self.r,self.theta_new*self.kappa_r))

            accept = (log_uniform < ll_new - ll_old)

            if accept:
                self.w[widx,:] = w_new
                self.z = z_new
                self.theta = theta_new
                self.sample_w_accept += 1
                self.particle_aggregation = agg_new
                self.one_per_particle_prob = one_per_particle_new

    def sample_u_posteriors(self,data_present = False):
        ## update particle centers using Metropolis-Hastings moves

        ## sample from Gaussian kernel
        proposals = rng.multivariate_normal(np.zeros(self.D), math.pow(self.u_std_tune*self.nu_u,2.0)*np.eye(self.D),self.numParticles)
        u_proposals = self.u + proposals
        # the priors are independent Gaussians, so we can vectorize this easily
        u_prior_old_loglikes = np.sum(logNormalLike(self.u,0,self.nu_u),axis=1)
        u_prior_new_loglikes = np.sum(logNormalLike(u_proposals,0,self.nu_u),axis=1)

        ## sample probabilities for acceptance
        log_uniforms = np.log(rng.random(self.numParticles))

        m_particle_vol = self.particle_vol(self.mlambda,self.D)
        total_particle_vol = np.sum(m_particle_vol)

        for uidx, (u_new, log_uniform, u_prior_old_loglike, u_prior_new_loglike) in enumerate(zip(u_proposals,log_uniforms, u_prior_old_loglikes, u_prior_new_loglikes)):
            # compute new z values
            z_new = self.calc_z(self.w, u_new, self.mlambda[uidx])
            theta_prime_new = self.omega*z_new + self.smooth_theta
            theta_new = theta_prime_new/np.sum(theta_prime_new)
            one_per_particle_new = self.min_one_per_particle_prob(z_new)
            agg_new = self.particle_aggregation - self.theta[uidx,:]*m_particle_vol[uidx]/total_particle_vol + theta_new*m_particle_vol[uidx]/total_particle_vol

            ll_old = u_prior_old_loglike + np.log(self.one_per_particle_prob[uidx]) + np.sum((self.beta*self.pi-1.0)*np.log(self.particle_aggregation))
            ll_new = u_prior_new_loglike + np.log(one_per_particle_new) + np.sum((self.beta*self.pi-1.0)*np.log(agg_new))

            if data_present is True:
                ## need to account for data in posterior under the Dirichlet Multinomial
                ll_old += logDirichletMultinomial(self.r,self.theta[uidx,:]*self.kappa_r)
                ll_new += logDirichletMultinomial(self.r,theta_new*self.kappa_r)

            accept = (log_uniform < ll_new - ll_old)

            if accept:
                self.u[uidx,:] = u_new
                self.z[uidx,:] = z_new
                self.theta[uidx,:] = theta_new
                self.sample_u_accept += 1
                self.particle_aggregation = agg_new
                self.one_per_particle_prob[uidx] = one_per_particle_new

    def sample_lambda_posteriors(self,data_present = False):
        ## update particle radii using Metropolis-Hastings moves

        ## sample from Gaussian kernel
        proposals = rng.normal(0,self.lambda_std_tune,self.numParticles)
        lambda_proposals = np.exp(np.log(self.mlambda) + proposals)
        lambda_prior_old_loglikes = logGammaLike(self.mlambda,self.eta_lambda1,self.eta_lambda2)
        lambda_prior_new_loglikes = logGammaLike(lambda_proposals,self.eta_lambda1,self.eta_lambda2)

        ## sample probabilities for acceptance
        log_uniforms = np.log(rng.random(self.numParticles))

        m_particle_vol = self.particle_vol(self.mlambda,self.D)
        total_particle_vol = np.sum(m_particle_vol)

        for lidx, (lambda_new, log_uniform, lambda_prior_old_loglike, lambda_prior_new_loglike) in enumerate(zip(lambda_proposals,log_uniforms, lambda_prior_old_loglikes, lambda_prior_new_loglikes)):
            # compute new z values
            z_new = self.calc_z(self.w, self.u[lidx], lambda_new)
            theta_prime_new = self.omega*z_new + self.smooth_theta
            theta_new = theta_prime_new/np.sum(theta_prime_new)
            one_per_particle_new = self.min_one_per_particle_prob(z_new)
            particle_vol_new = self.particle_vol(lambda_new,self.D)
            total_particle_vol_new = total_particle_vol - m_particle_vol[lidx] + particle_vol_new

            agg_new = (self.particle_aggregation*total_particle_vol - self.theta[lidx,:]*m_particle_vol[lidx] + theta_new*particle_vol_new)/total_particle_vol_new

            ll_old = lambda_prior_old_loglike + np.log(self.one_per_particle_prob[lidx]) + np.sum((self.beta*self.pi-1.0)*np.log(self.particle_aggregation))
            ll_new = lambda_prior_new_loglike + np.log(one_per_particle_new) + np.sum((self.beta*self.pi-1.0)*np.log(agg_new))

            if data_present is True:
                ## need to account for data in posterior under the Dirichlet Multinomial
                ll_old += logDirichletMultinomial(self.r,self.theta[uidx,:]*self.kappa_r)
                ll_new += logDirichletMultinomial(self.r,theta_new*self.kappa_r)

            accept = (log_uniform < ll_new - ll_old)

            if accept:
                self.z[lidx,:] = z_new
                self.theta[lidx,:] = theta_new
                self.mlambda[lidx] = lambda_new
                self.particle_aggregation = agg_new
                self.one_per_particle_prob[lidx] = one_per_particle_new
                total_particle_vol = total_particle_vol_new
                m_particle_vol[lidx] = particle_vol_new
                self.sample_lambda_accept += 1

#model = SpatialEmbed(10,50,3)
#model.generateSynthData()
#np.savetxt('bulk_reads.csv', model.q, delimiter='\t')
#np.savetxt('particle_reads.csv', model.r, delimiter='\t')
bulk_reads = np.loadtxt('bulk_reads.csv', delimiter='\t')
particle_reads = np.loadtxt('particle_reads.csv', delimiter='\t')
model = SpatialEmbed(10,50,3,bulk_reads,particle_reads)
print("pi =",model.pi)
print("omega =",model.omega)
print("particle_aggregation =",model.particle_aggregation)
print(model.q)
for i in range(0,1000):
    model.sample_omega_posteriors()
print("new omega = ",model.omega)
print("new particle_aggregation = ",model.particle_aggregation)
print(model.sample_omega_accept/1000)
