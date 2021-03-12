from __future__ import print_function
import torch
import torch.distributions as tdist
from mpl_toolkits.mplot3d import Axes3D
import numpy
from stepapprox import unitboxcar
import plotly.graph_objects as go
from plotvolume import soft_ellipse

#def make_sphere(ax,x,y,z,r):

#    detail_level = 32
#    u = numpy.linspace(0, 2 * numpy.pi, detail_level)
#    v = numpy.linspace(0, numpy.pi, detail_level)
#    X = x + r * numpy.outer(numpy.cos(u), numpy.sin(v))
#    Y = y + r * numpy.outer(numpy.sin(u), numpy.sin(v))
#    Z = z + r * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

#    ax.plot_surface(X, Y, Z, alpha=0.3)

# set up model to generate data

# embedding dimension
D = 3

# number of OTUs
O = 50

# sample embedding of OTUs
w = torch.zeros(O,D)
w_prior = tdist.MultivariateNormal(torch.zeros(D), torch.eye(D))

for o in range(0,O):
    w[o,:] = w_prior.sample()

# hyperparameters for particle radius
#eta_1 = 1.0
#eta_2 = 2.0
#rad_prior = tdist.Gamma(torch.tensor([eta_1]), torch.tensor([eta_2]))
#rad = rad_prior.sample()*4.0
#rad = rad * rad
mu_rad = numpy.log(1.0)
mu_std = 1.0
rad_prior = tdist.LogNormal(torch.tensor([mu_rad]),torch.tensor([mu_std]))
#rad = rad_prior.sample()
rad = torch.tensor([1.0])

# sample particle center
mu_prior = tdist.MultivariateNormal(torch.zeros(D), torch.eye(D))
mu = mu_prior.sample()

print('mu=',mu)
print('rad=',rad)

# sample indicator as to whether OTUs occur in the particle
#z = torch.zeros(O,1,dtype=torch.int64)
zr = torch.zeros(O,1,dtype=torch.float64)

# annealing parameter for unit step approximation
approx = 2

for o in range(0,O):
    p = mu-w[o,:]
    p = torch.pow(p,2.0)/rad
    p = (torch.sum(p)).sqrt()
    #p = torch.sum(0.5*p)
    #p = torch.exp(-p)
    zr[o] = unitboxcar(p, 0.0, 2.0, approx)
    #z[o] = tdist.Categorical(torch.tensor([1.0-p,p])).sample()

print(zr)

fig = go.Figure()
soft_ellipse(fig,mu,torch.ones(3)*rad.sqrt(),approx,rsize=3)

fig = fig.add_trace(go.Scatter3d(
    x = w[:,0],
    y = w[:,1],
    z = w[:,2],
    mode = 'markers',
    marker=dict(
        cmin = 0.0,
        cmax = 1.0,
        color = zr.flatten(),
        line=dict(width=2,color='Black')
    )
))

#soft_ellipse(fig,mu,torch.ones(3)*rad.sqrt(),approx,rsize=3)
fig.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#make_sphere(ax,mu[0].numpy(),mu[1].numpy(),mu[2].numpy(), numpy.sqrt(rad.numpy()))

#for o in range(0,O):
#    if z[o] == 1:
#        ax.scatter3D(w[o,0], w[o,1], w[o,2],c='b',marker='v')
#    else:
#        ax.scatter3D(w[o,0], w[o,1], w[o,2],c='r',marker='o')

#ax.scatter3D(mu[0],mu[1],mu[2],c='r')

#plt.show()
