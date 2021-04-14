import numpy as np
import scipy.stats
import math
import scipy.special as sc
import matplotlib.pyplot as plt
import pydot

from numpy.random import default_rng

rng = default_rng(seed=2)

def sample_BinaryConcrete(alpha0,mlambda):
    u = rng.random(size=alpha0.size)
    g = np.log(u) - np.log(1.0-u)
    alpha = alpha0/(1-alpha0)
    return 1.0/(1.0+np.exp(-(np.log(alpha) + g)/mlambda))

def chol_inv(X):
    # invert matrix via Cholesky decomposition
    # note: matrix must be positive-definite, but this function does no check
    c = np.linalg.inv(np.linalg.cholesky(X))
    return np.dot(c.T, c)

class TreeNode:
    def __init__(self,idx,depth):
        self.mydepth = 0
        self.idx = idx
        self.keep_going = 0.0
        self.value = 0.0
        self.value_inc = 0.0
        self.left_point = 0.0
        self.right_point = 0.0
        self.split_param = 0.0
        self.split_proportion = 0.0
        self.split_point = 0.0

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
        # no children, nothing to do
        if self.is_leaf:
            return

        delta = self.right_point-self.left_point
        self.split_point = self.split_proportion*delta + self.left_point
        nodes[self.left()].left_point = self.left_point
        nodes[self.left()].right_point = self.split_point

        nodes[self.right()].right_point = self.right_point
        nodes[self.right()].left_point = self.split_point

    def print_interval(self):
        return "[" + str(self.left_point) + ", " + str(self.right_point) + "]"

    def sample_split_from_prior(self):
        # no split at leaf
        if self.is_leaf:
            return

        self.split_param = rng.normal(0,1)
        self.split_proportion = 1/(1+np.exp(self.split_param))

    def sample_keep_going_from_prior(self,keep_going_rate,concrete_temp):
        # always stop if a leaf
        if self.is_leaf:
            return

        #prob_stop = (1-np.ones(1)*math.pow(keep_going_rate,self.mydepth))
        prob_keep_going = np.ones(1)*math.pow(keep_going_rate,self.mydepth)
        self.keep_going = sample_BinaryConcrete(prob_keep_going,concrete_temp)

class RegularizerFunction:
    def __init__(self,inducing_points):
        self.inducing_points = inducing_points

class LinearRegularizer(RegularizerFunction):
    def __init__(self,inducing_points,slope_init):
        super().__init__(inducing_points)
        self.slope = slope_init

    def eval(self):
        return self.inducing_points*self.slope

class UnivariateRegressionTree:
    def __init__(self,depth,interval,regularizer_function=None,data=None):
        # structure is balanced binary tree
        self.depth = depth
        self.num_nodes = int(math.pow(2,self.depth+1)-1)
        self.nodes = list()
        self.regularizer_function = regularizer_function
        self.data = data

        # priors for values on tree
        #self.sigma_v0 = 10.0
        self.sigma_v0 = 1000.0
        self.mu_v0 = 100.0
        #self.sigma_v = 50
        self.sigma_v = 1000.0

        # set up prior for stopping in tree
        self.keep_going_rate = 0.50
        # lower temp approximates a draw from a Bernoulli
        self.concrete_temp = 0.01

        # parameter for sharpness of cuts down tree
        self.alpha_rho = 1.0

        # prior for measurement noise (assumed to be estimated from technical replicates)
        self.sigma_data = 10.0

        # parameters for functional regularization
        self.function_reg_constant = 1.0
        self.zeta = 0.001
        if self.data is not None:
            if self.regularizer_function is not None:
                self.function_reg_constant = len(self.data)/len(self.regularizer_function.inducing_points)

        for i in range(0,self.num_nodes):
            node = TreeNode(i,depth)
            self.nodes.append(node)

        for node in self.nodes:
            if node.is_leaf is False:
                # push depth
                self.nodes[node.left()].mydepth = node.mydepth + 1
                self.nodes[node.right()].mydepth = node.mydepth + 1

        self.nodes[0].left_point = interval[0]
        self.nodes[0].right_point = interval[1]

    def sample_values_from_prior(self):
        # sample initial value
        self.nodes[0].value = rng.normal(self.mu_v0,self.sigma_v0)

        interval_size = self.nodes[0].right_point-self.nodes[0].left_point

        # sample values and stopping down the tree
        for i in range(1,self.num_nodes):
            nidx = self.nodes[i].parent()
            interval_proportion = (self.nodes[i].right_point-self.nodes[i].left_point)/interval_size
            print(interval_proportion)
            self.nodes[i].value_inc = rng.normal(0,self.sigma_v*interval_proportion)
            self.nodes[i].value = self.nodes[nidx].value + self.nodes[i].value_inc
            self.nodes[i].sample_keep_going_from_prior(self.keep_going_rate,self.concrete_temp)

        self.nodes[0].keep_going = 1.0

    def sample_splits_from_prior(self):
        for i in range(0,self.num_nodes):
            self.nodes[i].sample_split_from_prior()

    def calc_intervals(self):
        for i in range(0,self.num_nodes):
            self.nodes[i].calc_child_intervals(self.nodes)

    def eval_tree(self,x):
        # x is the vector of values to evaluate the tree

        # in more efficient implementation, could pre-allocate and re-use arrays

        # 'soft decisions' at cut points
        rho = np.zeros((x.size,self.num_nodes))

        # probability of reaching node
        q = np.ones((x.size,self.num_nodes))

        self.calc_tree_probs_below(x, rho, q, 0)

        # calc values
        v = np.zeros(self.num_nodes)
        keep_going = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            v[i] = self.nodes[i].value
            keep_going[i] = self.nodes[i].keep_going
        return np.matmul(q*(1.0-keep_going), v)

    #def calc_values_below(self,q,v,start_idx):
    #    for i in range(start_idx, self.num_nodes):
    #        v += q[i,:] * (1-self.nodes[i].keep_going) * self.nodes[i].value

    def calc_tree_probs_below(self,x,rho,q,start_idx):
        # x is the vector of values to evaluate the tree
        # rho is a # nodes X data size matrix for storing 'soft decisions' at cut points
        # q is a # nodes X data size matrix for storing probabilities of reaching nodes
        # start_idx is node index to start evaluation

        # push probabilities down the tree
        for i in range(start_idx,self.num_nodes):
            node = self.nodes[i]
            if node.is_leaf is False:
                rho[:,i] = 1.0/(1.0+np.exp(-self.alpha_rho*(x-node.split_point)))
                q[:,node.left()] = (1-rho[:,i])*q[:,i]*node.keep_going
                q[:,node.right()] = rho[:,i] * q[:,i] * node.keep_going

    def sample_posterior(self,num_iters):
        if self.data is not None:
            # 'soft decisions' at cut points
            rho_data = np.zeros((self.data.size, self.num_nodes))
            # probability of reaching node
            q_data = np.ones((self.data.size, self.num_nodes))
            self.calc_tree_probs_below(self.data, rho_data, q_data, 0)
            var_data = np.ones(self.data.size)*math.pow(self.sigma_data,2.0)

        if self.regularizer_function is not None:
            # 'soft decisions' at cut points
            rho_fn = np.zeros((self.regularizer_function.inducing_points.size, self.num_nodes))
            # probability of reaching node
            q_fn = np.ones((self.regularizer_function.inducing_points.size, self.num_nodes))
            self.calc_tree_probs_below(self.regularizer_function.inducing_points, rho_fn, q_fn, 0)
            y_fn = self.regularizer_function.eval()
            var_fn = np.ones(self.regularizer_function.inducing_points.size)*math.pow(self.sigma_data,2.0)/(self.zeta*self.function_reg_constant)

        interval_size = self.nodes[0].right_point - self.nodes[0].left_point
        v_prior_mat = np.zeros((self.num_nodes,self.num_nodes))
        v_prior_var = np.zeros(self.num_nodes)
        y_v_prior = np.zeros(self.num_nodes)
        # prior on first value
        v_prior_mat[0,0] = 1.0
        v_prior_var[0] = math.pow(self.sigma_v0,2.0)
        y_v_prior[0] = self.mu_v0
        # prior on adjacent values
        for i in range(1,self.num_nodes):
            v_prior_mat[i,i-1] = -1.0
            v_prior_mat[i,i] = 1.0
            interval_proportion = (self.nodes[i].right_point - self.nodes[i].left_point) / interval_size
            v_prior_var[i] = math.pow(self.sigma_v * interval_proportion,2.0)

        v = np.zeros(self.num_nodes)
        keep_going = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            v[i] = self.nodes[i].value
            keep_going[i] = self.nodes[i].keep_going

        if self.data is None or self.regularizer_function is None:
            if (self.data is not None):
                Y = self.data
                VAR = var_data
                X = q_data*(1.0-keep_going)
            else:
                Y = y_fn
                VAR = var_fn
                X = q_fn * (1.0 - keep_going)
        else:
            Y = np.concatenate([self.data,y_fn])
            VAR = np.concatenate([var_data,var_fn])
            X = np.concatenate([q_data*(1.0-keep_going), q_fn * (1.0 - keep_going)])

        Y = np.concatenate([Y,y_v_prior])
        VAR = np.concatenate([VAR, v_prior_var])
        X = np.concatenate([X, v_prior_mat])

        v = self.sample_value_posterior(X,Y,VAR)
        return np.matmul(X,v)

    def sample_value_posterior(self,X,Y,V):
        # the conditional posterior for the values reduces
        # to a Bayesian linear regression
        sigma = chol_inv(np.matmul(np.matmul(X.T,np.diag(1/V)),X))
        mu = np.matmul(sigma,np.matmul(np.matmul(X.T,np.diag(1/V)),Y))
        return rng.multivariate_normal(mu,sigma)

    def visualize_tree(self):
        graph = pydot.Dot(graph_type='graph')
        self.visualize_tree_recurse(self.nodes[0],graph)
        print(graph.to_string())
        graph.write_png("tree.png")

    def visualize_tree_recurse(self,node,graph):
        #viznode = pydot.Node(str(node.idx),label=node.print_interval()+" val=" + str(node.value),shape='box')
        if node.keep_going <= 0.95:
            interval = '[' + f'{node.left_point:.2f}' + ', ' + f'{node.right_point:.2f}' + ']'
            viznode = pydot.Node(str(node.idx),label=interval + '\n' + "val=" + f'{node.value:.2f}',shape='box',color='red')
        else:
            viznode = pydot.Node(str(node.idx),label=' ',shape='box', color='blue')
        graph.add_node(viznode)
        if node.keep_going > 0.95:
            edge_left = pydot.Edge(str(node.idx),str(node.left()),label='<'+f'{node.split_point:.2f}')
            graph.add_edge(edge_left)
            edge_right = pydot.Edge(str(node.idx),str(node.right()))
            graph.add_edge(edge_right)
            self.visualize_tree_recurse(self.nodes[node.left()],graph)
            self.visualize_tree_recurse(self.nodes[node.right()],graph)

x = np.linspace(50,500,50)
linfn = LinearRegularizer(x,1.0)
fn_val = linfn.eval()
tree = UnivariateRegressionTree(3,[50,500],linfn)
tree.sample_splits_from_prior()
tree.calc_intervals()
tree.sample_values_from_prior()

fig, ax = plt.subplots()
for i in range(0,10):
    v = tree.sample_posterior(1)
    ax.plot(x,v[0:50])
ax.plot(x,fn_val)
plt.show()

#for i in range(0,len(tree.nodes)):
#    print("node ",i," int ", tree.nodes[i].print_interval(), ' val=',tree.nodes[i].value, ' keep_going=',tree.nodes[i].keep_going)
#tree.visualize_tree()

#v = tree.eval_tree(x)
#fig, ax = plt.subplots()
#ax.plot(x,v)
#plt.show()
