import plotly.graph_objects as go
import numpy as np
from stepapprox import unitboxcar

def soft_ellipse(fig,mu,rad,approx,rsize=3):

    amin = -rsize
    amax = rsize
    X, Y, Z = np.mgrid[amin:amax:50j, amin:amax:50j, amin:amax:50j]
    mu_X = mu[0]
    mu_Y = mu[1]
    mu_Z = mu[2]
    rad_X = np.power(rad[0],2.0)
    rad_Y = np.power(rad[1],2.0)
    rad_Z = np.power(rad[2],2.0)

    x = np.power(mu_X - X,2.0)/rad_X
    y = np.power(mu_Y - Y,2.0)/rad_Y
    z = np.power(mu_Z - Z,2.0)/rad_Z

    values = np.sqrt(x + y + z)

    values2 = unitboxcar(values, 0.0, 2.0, approx)

    #fig = go.Figure(data=go.Volume(
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values2.flatten(),
        isomin=0.01,
        isomax=1.0,
        opacity=0.05, # needs to be small to see through all surfaces
        surface_count=25, # needs to be a large number for good volume rendering
        ))
    #fig.show()

#soft_ellipse(np.ones(3),np.ones(3)*2.0,4.0,rsize=3)
