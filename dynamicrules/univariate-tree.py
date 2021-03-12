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

class UnivariateRegressionTree:
    def __init__(self,depth,interval):
        # structure is balanced binary tree
        self.depth = depth
        self.num_nodes = int(math.pow(2,self.depth+1)-1)
        self.nodes = list()

        # damps values going down the tree
        self.damp_rate = 0.0

        # priors for values on tree
        self.sigma_v0 = 10.0
        self.mu_v0 = 100.0
        self.sigma_v = 50

        # set up prior for stopping in tree
        # this rate will yield an expected value of 1 per level
        self.keep_going_rate = 0.50
        # lower temp approximates a draw from a Bernoulli
        self.concrete_temp = 0.01

        # parameter for sharpness of cuts down tree
        self.alpha_rho = 1.0

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
            self.nodes[i].value = self.nodes[nidx].value + self.nodes[i].value_inc*np.exp(-self.damp_rate)
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
        rho = np.zeros((self.num_nodes,x.size))

        # probability of reaching node
        q = np.ones((self.num_nodes,x.size))

        # push probabilities down the tree
        for i in range(0,self.num_nodes):
            node = self.nodes[i]
            if node.is_leaf is False:
                rho[i,:] = 1.0/(1.0+np.exp(-self.alpha_rho*(x-node.split_point)))
                q[node.left(),:] = (1-rho[i,:])*q[i,:]*node.keep_going
                q[node.right(),:] = rho[i, :] * q[i, :] * node.keep_going

        # calc values
        v = np.zeros(x.shape)
        for i in range(0, self.num_nodes):
            v = v + q[i,:] * (1-self.nodes[i].keep_going) * self.nodes[i].value

        return v

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

tree = UnivariateRegressionTree(3,[50,500])
tree.sample_splits_from_prior()
tree.calc_intervals()
tree.sample_values_from_prior()
for i in range(0,len(tree.nodes)):
    print("node ",i," int ", tree.nodes[i].print_interval(), ' val=',tree.nodes[i].value, ' keep_going=',tree.nodes[i].keep_going)
tree.visualize_tree()
x = np.linspace(50,500,50)
v = tree.eval_tree(x)
fig, ax = plt.subplots()
ax.plot(x,v)
plt.show()
#print(sample_BinaryConcrete(np.ones(4)*0.5,0.5))
