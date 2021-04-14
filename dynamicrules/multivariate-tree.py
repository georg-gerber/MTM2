import numpy as np
import math
import matplotlib.pyplot as plt
import pydot
import scipy as sci

from numpy.random import default_rng

rng = default_rng(seed=1)

def sample_Concrete(alpha,mlambda):
    u = rng.random(size=alpha.shape)
    g = -np.log(-np.log(u))
    X = np.exp((np.log(alpha) + g)/mlambda)
    if (alpha.ndim > 1):
        return X.T/np.sum(X,axis=1)
    else:
        return X/sum(X)

def sample_BinaryConcrete(alpha0,mlambda):
    u = rng.random()
    g = np.log(u) - np.log(1.0-u)
    alpha = alpha0/(1-alpha0)
    return 1.0/(1.0+np.exp(-(np.log(alpha) + g)/mlambda))

class TreeNode:
    def __init__(self,idx,depth,num_covariates):
        self.num_covariates = num_covariates
        self.mydepth = 0
        self.idx = idx
        self.keep_going = 0.0
        self.select_covariates = np.zeros(num_covariates)
        self.value = 0.0
        self.value_inc = 0.0
        self.left_point = np.zeros(num_covariates)
        self.right_point = np.zeros(num_covariates)
        self.split_proportion = np.zeros(num_covariates)
        self.split_point = np.zeros(num_covariates)

        if self.left() >= math.pow(2,depth+1)-1:
            self.is_leaf = True
        else:
            self.is_leaf = False

    def left(self):
        return 2*self.idx + 1

    def right(self):
        return 2*self.idx + 2

    def parent(self):
        return int((self.idx-1)/2)

    def calc_child_intervals(self,nodes):
        # if no children, nothing to do
        if self.is_leaf:
            return

        delta = self.right_point-self.left_point
        #self.split_point = self.split_proportion*delta*self.select_covariates + self.left_point
        self.split_point = self.split_proportion * delta + self.left_point
        nodes[self.left()].left_point = self.left_point
        #nodes[self.left()].right_point = self.split_point
        nodes[self.left()].right_point = self.split_point*self.select_covariates + \
                                         self.right_point*(1.0-self.select_covariates)

        nodes[self.right()].right_point = self.right_point
        #nodes[self.right()].left_point = self.split_point
        nodes[self.right()].left_point = self.split_point*self.select_covariates + \
                                         self.left_point*(1.0-self.select_covariates)

    def print_interval(self):
        select_var = np.argmax(self.select_covariates)
        return "[" + str(self.left_point[select_var]) + ", " + str(self.right_point[select_var]) + "]"

    def sample_split_from_prior(self):
        # no split at leaf
        if self.is_leaf:
            return
        self.split_proportion = rng.beta(1.0,1.0,self.num_covariates)

    def sample_keep_going_from_prior(self,prob_keep_going,concrete_temp):
        # always stop if a leaf
        if self.is_leaf:
            return

        self.keep_going = sample_BinaryConcrete(prob_keep_going,concrete_temp)

    def sample_select_covariates_from_prior(self,psi,concrete_temp):
        self.select_covariates = sample_Concrete(psi,concrete_temp)

class MultivariateRegressionTree:
    def __init__(self,depth,min_range,max_range,data=None):
        # structure is balanced binary tree
        # min_range is a vector with minimum values for covariates
        # max_range is a vector with maximum values for covariates

        self.depth = depth
        self.num_nodes = int(math.pow(2,self.depth+1)-1)
        self.nodes = list()
        self.data = data
        self.num_covariates = len(min_range)

        # priors for values on tree
        self.sigma_v0 = 10.0
        self.mu_v0 = 100.0
        self.sigma_v = 50

        # set up prior for 'keep going' in tree
        self.mlambda1 = 0.90
        self.mlambda2 = 0.75
        #self.mlambda2 = 1.0

        # prior for selecting variables
        self.alpha_psi = 1.0 # value of 1.0 is uniform
        self.psi = np.zeros(self.num_covariates)

        # lower temp approximates a draw from a Multinomial
        self.concrete_temp = 0.005

        # parameter for sharpness of cuts down tree
        self.alpha_rho = 1.0

        # prior for measurement noise (assumed to be estimated from technical replicates)
        self.sigma_data = 10.0

        for i in range(0,self.num_nodes):
            node = TreeNode(i,depth,self.num_covariates)
            self.nodes.append(node)

        for node in self.nodes:
            if node.is_leaf is False:
                # push depth
                self.nodes[node.left()].mydepth = node.mydepth + 1
                self.nodes[node.right()].mydepth = node.mydepth + 1

        self.nodes[0].left_point = min_range
        self.nodes[0].right_point = max_range

    def sample_covariate_select_from_prior(self):
        self.psi = rng.dirichlet(np.ones(self.num_covariates)*self.alpha_psi/self.num_covariates)
        for i in range(0,self.num_nodes):
            self.nodes[i].sample_select_covariates_from_prior(self.psi,self.concrete_temp)

    def sample_values_from_prior(self):
        # sample initial value
        self.nodes[0].value = rng.normal(self.mu_v0,self.sigma_v0)

        # sample values and stopping down the tree
        for i in range(1,self.num_nodes):
            nidx = self.nodes[i].parent()
            self.nodes[i].value_inc = rng.normal(0,self.sigma_v)
            self.nodes[i].value = self.nodes[nidx].value + self.nodes[i].value_inc
            keep_going_prob = self.mlambda1*math.pow(1+self.nodes[i].mydepth,-self.mlambda2)
            self.nodes[i].sample_keep_going_from_prior(keep_going_prob,self.concrete_temp)

        self.nodes[0].keep_going = 1.0

    def sample_splits_from_prior(self):
        for i in range(0,self.num_nodes):
            self.nodes[i].sample_split_from_prior()

    def calc_intervals(self):
        for i in range(0,self.num_nodes):
            self.nodes[i].calc_child_intervals(self.nodes)

    def visualize_tree(self):
        graph = pydot.Dot(graph_type='graph')
        self.visualize_tree_recurse(self.nodes[0],self.nodes[0].select_covariates,graph)
        #print(graph.to_string())
        graph.write_png("tree.png")

    def visualize_tree_recurse(self,node,select_covariates,graph):
        #viznode = pydot.Node(str(node.idx),label=node.print_interval()+" val=" + str(node.value),shape='box')
        select_var = np.argmax(select_covariates)
        if node.keep_going <= 0.95:
            interval = '[' + f'{node.left_point[select_var]:.2f}' + ', ' + f'{node.right_point[select_var]:.2f}' + ']'
            viznode = pydot.Node(str(node.idx),label=interval + '\n' + "val=" + f'{node.value:.2f}',shape='box',color='red')
        else:
            viznode = pydot.Node(str(node.idx),label=' ',shape='box', color='blue')
        graph.add_node(viznode)
        if node.keep_going > 0.95:
            select_var_child = np.argmax(node.select_covariates)
            edge_left = pydot.Edge(str(node.idx),str(node.left()),label='var ' + str(select_var_child) + ' <'+f'{node.split_point[select_var_child]:.2f}')
            graph.add_edge(edge_left)
            edge_right = pydot.Edge(str(node.idx),str(node.right()))
            graph.add_edge(edge_right)
            self.visualize_tree_recurse(self.nodes[node.left()],node.select_covariates,graph)
            self.visualize_tree_recurse(self.nodes[node.right()],node.select_covariates,graph)

    def eval_tree(self,x):
        # x is the vector of values to evaluate the tree

        # in more efficient implementation, could pre-allocate and re-use arrays

        # 'soft decisions' at cut points
        rho = np.zeros((self.num_nodes,x.shape[0],x.shape[1]))

        # probability of reaching node
        q = np.ones((self.num_nodes,x.shape[0]))

        self.calc_tree_probs_below(x, rho, q, 0)

        # calc values
        v = np.zeros(self.num_nodes)
        keep_going = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            v[i] = self.nodes[i].value
            keep_going[i] = self.nodes[i].keep_going
        return np.matmul(q.T*(1.0-keep_going), v)

    def calc_tree_probs_below(self,x,rho,q,start_idx):
        # x is the values to evaluate the tree at
        # rho is a 3D matrix # nodes X # data points X # variables for storing 'soft decisions' at cut points
        # q is a # nodes X data points matrix for storing probabilities of reaching nodes
        # start_idx is node index to start evaluation

        # push probabilities down the tree
        for i in range(start_idx,self.num_nodes):
            node = self.nodes[i]
            if node.is_leaf is False:
                rho[i,:] = 1.0/(1.0+np.exp(-self.alpha_rho*(x-node.split_point)))
                left_rho = np.sum((1.0-rho[i,:])*node.select_covariates,axis=1)
                q[node.left(),:] = left_rho*q[i,:]*node.keep_going
                right_rho = np.sum(rho[i,:]*node.select_covariates,axis=1)
                q[node.right(),:] = right_rho * q[i,:] * node.keep_going

tree = MultivariateRegressionTree(4,np.array([50,0]),np.array([500,1.0]))
tree.sample_covariate_select_from_prior()
tree.sample_splits_from_prior()
tree.calc_intervals()
tree.sample_values_from_prior()
tree.visualize_tree()

x1 = np.linspace(50,500,50)
x2 = np.linspace(0,1,50)
X1,X2 = np.meshgrid(x1,x2)
grid=np.array([X1.flatten(),X2.flatten()]).T

Y = tree.eval_tree(grid)

ax = plt.axes(projection='3d')
ax.plot_trisurf(grid[:,0],grid[:,1],Y,cmap='viridis', edgecolor='none')
plt.show()

#x1i = np.linspace(np.min(x1),np.max(x1),100)
#x2i = np.linspace(np.min(x2),np.max(x2),100)
#zi = sci.interpolate.griddata(grid[0,:],grid[])