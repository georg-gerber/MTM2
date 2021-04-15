import torch
import torch.distributions
import math
import matplotlib.pyplot as plt
import pydot
import copy

torch.manual_seed(0)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def sample_BinaryConcrete(alpha0,mlambda):
    #u = torch.ones(alpha0.shape).uniform_(0,1)
    #g = torch.log(u) - torch.log(1.0-u)
    l = sample_StandardLogistic(alpha0.shape)
    alpha = alpha0/(1-alpha0)
    return l, (1.0/(1.0+torch.exp(-(torch.log(alpha) + l)/mlambda)))

def sample_StandardLogistic(shape):
    # sample from a standard Logistic variable (location 0, scale 1)
    u = torch.ones(shape).uniform_(0, 1)
    return torch.log(u) - torch.log(1.0 - u)

def logpdf_StandardLogistic(l):
    return (-l - 2*(1.0+torch.exp(-l)))

class TreeNode:
    def __init__(self,idx,depth):
        self.mydepth = 0
        self.idx = idx
        self.keep_going_prob = 1.0
        self.keep_going = torch.tensor(0.0)
        self.keep_going_logistic = torch.tensor(0.0)

        self.value = torch.tensor(0.0)
        self.left_point = torch.tensor(0.0)
        self.right_point = torch.tensor(0.0)
        self.split_proportion = torch.tensor(0.0)
        self.split_point = torch.tensor(0.0)

        if self.left() >= math.pow(2,depth+1)-1:
            self.is_leaf = True
        else:
            self.is_leaf = False

    def clone_self(self):
        node = copy.copy(self)
        node.keep_going = self.keep_going.detach().clone()
        node.keep_going_logistic = self.keep_going_logistic.detach().clone()
        node.value = self.value.detach().clone()
        node.left_point = self.left_point.detach().clone()
        node.right_point = self.right_point.detach().clone()
        node.split_proportion = self.split_proportion.detach().clone()
        node.split_point = self.split_point.detach().clone()

        return node

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

    def calc_value_std(self,nodes,mrange,sigma_va,sigma_vb):
        parentidx = self.parent()
        midpoint = (self.right_point + self.left_point) / 2.0
        parent_midpoint = (nodes[parentidx].right_point + nodes[parentidx].left_point) / 2.0
        return (sigma_va + sigma_vb * torch.abs(parent_midpoint - midpoint) / mrange)

    def print_interval(self):
        return "[" + str(self.left_point) + ", " + str(self.right_point) + "]"

    def sample_split_from_prior(self):
        # no split at leaf
        if self.is_leaf:
            return

        self.split_proportion = torch.distributions.Beta(1.0, 1.0).sample()

    def sample_keep_going_from_prior(self,concrete_temp):
        # always stop if a leaf
        if self.is_leaf:
            return

        self.keep_going_logistic, self.keep_going = sample_BinaryConcrete(self.keep_going_prob, concrete_temp)

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
        self.mrange = interval[1]-interval[0]

        # priors for values on tree
        #self.sigma_v0 = 10.0
        self.sigma_v0 = torch.tensor(1000.0)
        self.mu_v0 = torch.tensor(100.0)
        #self.sigma_v = 50
        self.sigma_va = torch.tensor(10.0)
        self.sigma_vb = torch.tensor(500.0)

        # set up prior for 'keep going' in tree
        self.mlambda1 = torch.tensor(0.90)
        self.mlambda2 = torch.tensor(0.75)

        # lower temp approximates a draw from a Bernoulli
        self.concrete_temp = 0.001
        #self.concrete_temp = 0.01
        #self.concrete_temp = 0.05

        # parameter for sharpness of cuts down tree
        self.alpha_rho = torch.tensor(10.0)
        #self.alpha_rho = torch.tensor(0.5)
        #self.alpha_rho = torch.tensor(0.1)

        # prior for measurement noise (assumed to be estimated from technical replicates)
        self.sigma_data = torch.tensor(10.0)

        # parameters for functional regularization
        self.function_reg_constant = torch.tensor(1.0)
        #self.zeta = 0.001
        self.zeta = 0.1
        if self.data is not None:
            if self.regularizer_function is not None:
                self.function_reg_constant = torch.tensor(len(self.data)/len(self.regularizer_function.inducing_points))

        # MCMC tuning parameters
        self.split_proposal_tune = 5
        self.keep_going_logistic_proposal_tune = 1.0

        for i in range(0,self.num_nodes):
            node = TreeNode(i,depth)
            self.nodes.append(node)

        for node in self.nodes:
            if node.is_leaf is False:
                # push depth
                self.nodes[node.left()].mydepth = node.mydepth + 1
                self.nodes[node.right()].mydepth = node.mydepth + 1

        # compute probability to keep going at each node
        for i in range(0,self.num_nodes):
            self.nodes[i].keep_going_prob = self.mlambda1 * torch.pow(1 + self.nodes[i].mydepth, -self.mlambda2)

        self.nodes[0].left_point = torch.tensor(interval[0])
        self.nodes[0].right_point = torch.tensor(interval[1])

    def sample_values_from_prior(self):
        # sample initial value
        self.nodes[0].value = torch.distributions.Normal(self.mu_v0,self.sigma_v0).sample()

        # sample keep_going values down the tree
        for i in range(0,self.num_nodes):
            self.nodes[i].sample_keep_going_from_prior(self.concrete_temp)

        # sample values down the tree
        for i in range(1,self.num_nodes):
            parentidx = self.nodes[i].parent()
            #interval_proportion = (self.nodes[i].right_point-self.nodes[i].left_point)/self.range
            self.nodes[i].value = self.nodes[parentidx].value + torch.distributions.Normal(torch.tensor(0.0),self.nodes[i].calc_value_std(self.nodes,self.mrange,self.sigma_va,self.sigma_vb)).sample()
        self.nodes[0].keep_going = torch.tensor(1.0)

    def sample_splits_from_prior(self):
        for i in range(0, self.num_nodes):
            self.nodes[i].sample_split_from_prior()

    def calc_intervals(self):
        for i in range(0, self.num_nodes):
            self.nodes[i].calc_child_intervals(self.nodes)

    def eval_tree(self,x):
        # x is the vector of values to evaluate the tree

        # in more efficient implementation, could pre-allocate and re-use arrays

        # 'soft decisions' at cut points
        rho = torch.zeros((len(x),self.num_nodes))

        # probability of reaching node
        q = torch.ones((len(x),self.num_nodes))

        self.calc_tree_probs_below(x, rho, q, 0)

        # calc values
        v = torch.zeros(self.num_nodes)
        keep_going = torch.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            v[i] = self.nodes[i].value
            keep_going[i] = self.nodes[i].keep_going
        return torch.matmul(q*(1.0-keep_going), v)

    def calc_tree_probs_below(self,x,rho,q,start_idx):
        # x is the vector of values to evaluate the tree
        # rho is a # nodes X data size matrix for storing 'soft decisions' at cut points
        # q is a # nodes X data size matrix for storing probabilities of reaching nodes
        # start_idx is node index to start evaluation

        # push probabilities down the tree
        for i in range(start_idx,self.num_nodes):
            node = self.nodes[i]
            if node.is_leaf is False:
                rho[:,i] = 1.0/(1.0+torch.exp(-self.alpha_rho*(x-node.split_point)))
                q[:,node.left()] = (1-rho[:,i])*q[:,i]*node.keep_going
                q[:,node.right()] = rho[:,i] * q[:,i] * node.keep_going

    def visualize_tree(self,fname):
        graph = pydot.Dot(graph_type='graph')
        self.visualize_tree_recurse(self.nodes[0],graph)
        #print(graph.to_string())
        graph.write_png(fname)

    def visualize_tree_recurse(self,node,graph):
        #viznode = pydot.Node(str(node.idx),label=node.print_interval()+" val=" + str(node.value),shape='box')
        if node.keep_going <= 0.95:
        #if node.keep_going <= 1.0:
            interval = '[' + f'{node.left_point:.2f}' + ', ' + f'{node.right_point:.2f}' + ']'
            viznode = pydot.Node(str(node.idx),label=interval + '\n' + "val=" + f'{node.value:.2f}',shape='box',color='red')
        else:
            #viznode = pydot.Node(str(node.idx),label=' ',shape='box', color='blue')
            viznode = pydot.Node(str(node.idx), label="val=" + f'{node.value:.2f}', shape='box', color='blue')
        graph.add_node(viznode)
        if node.keep_going > 0.95:
        #if node.keep_going > 0.50:
            edge_left = pydot.Edge(str(node.idx),str(node.left()),label='<'+f'{node.split_point:.2f}')
            graph.add_edge(edge_left)
            edge_right = pydot.Edge(str(node.idx),str(node.right()))
            graph.add_edge(edge_right)
            self.visualize_tree_recurse(self.nodes[node.left()],graph)
            self.visualize_tree_recurse(self.nodes[node.right()],graph)

    def sample_posterior(self,num_iters):
        rho_data = None
        q_data = None
        rho_fn = None
        q_fn = None

        if self.data is not None:
            # 'soft decisions' at cut points
            rho_data = torch.zeros((self.data.size, self.num_nodes))
            # probability of reaching node
            q_data = torch.ones((self.data.size, self.num_nodes))
            self.calc_tree_probs_below(self.data, rho_data, q_data, 0)
            var_data = torch.ones(self.data.size)*math.pow(self.sigma_data,2.0)

        if self.regularizer_function is not None:
            # 'soft decisions' at cut points
            rho_fn = torch.zeros((len(self.regularizer_function.inducing_points), self.num_nodes))
            # probability of reaching node
            q_fn = torch.ones((len(self.regularizer_function.inducing_points), self.num_nodes))
            self.calc_tree_probs_below(self.regularizer_function.inducing_points, rho_fn, q_fn, 0)
            y_fn = self.regularizer_function.eval()
            var_fn = torch.ones(len(self.regularizer_function.inducing_points))*math.pow(self.sigma_data,2.0)/(self.zeta*self.function_reg_constant)

        v_prior_mat = torch.zeros((self.num_nodes,self.num_nodes))
        v_prior_var = torch.zeros(self.num_nodes)
        y_v_prior = torch.zeros(self.num_nodes)
        # prior on first value
        v_prior_mat[0,0] = 1.0
        v_prior_var[0] = math.pow(self.sigma_v0,2.0)
        y_v_prior[0] = self.mu_v0
        # prior on adjacent values
        for i in range(1,self.num_nodes):
            parentidx = self.nodes[i].parent()
            v_prior_mat[i,parentidx] = -1.0
            v_prior_mat[i,i] = 1.0
            #interval_proportion = (self.nodes[i].right_point - self.nodes[i].left_point) / interval_size
            v_prior_var[i] = math.pow(self.nodes[i].calc_value_std(self.nodes,self.mrange,self.sigma_va,self.sigma_vb),2.0)
            # self.calc_value_std(self,self.nodes,self.mrange,self.sigma_va,self.sigma_vb)

        v = torch.zeros(self.num_nodes)
        keep_going = torch.zeros(self.num_nodes)
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
            Y = torch.cat([self.data,y_fn])
            VAR = torch.cat([var_data,var_fn])
            X = torch.cat([q_data*(1.0-keep_going), q_fn * (1.0 - keep_going)])

        Y_total = torch.cat([Y,y_v_prior])
        VAR_total = torch.cat([VAR, v_prior_var])
        X_total = torch.cat([X, v_prior_mat])

        # sample values
        v = self.sample_value_posterior_GIBBS(X_total,Y_total,VAR_total)
        for i in range(0, self.num_nodes):
            self.nodes[i].value = v[i]

        # sample splits
        for node_idx in range(0,self.num_nodes):
            if self.nodes[node_idx].is_leaf is False:
                accept, new_tree, keep_going_new, v_prior_var_new, rho_new_data, q_new_data, rho_new_fn, q_new_fn, X_new = self.sample_split_or_keep_going_posterior('split',node_idx,keep_going,Y_total,v,X,X_total,VAR,VAR_total,v_prior_mat,rho_data,q_data,rho_fn,q_fn)
                #print(accept)
                if accept:
                    # update tree
                    self.nodes = new_tree

                    # update matrices
                    rho_data = rho_new_data
                    q_data = q_new_data

                    rho_fn = rho_new_fn
                    q_fn = q_new_fn

                    X = X_new
                    X_total = torch.cat([X, v_prior_mat])
                    VAR_total = torch.cat([VAR, v_prior_var_new])

                v = self.sample_value_posterior_GIBBS(X_total, Y_total, VAR_total)
                for i in range(0, self.num_nodes):
                    self.nodes[i].value = v[i]

        # sample keep_going
        for node_idx in range(0,self.num_nodes):
            if self.nodes[node_idx].is_leaf is False:
                accept, new_tree, keep_going_new, v_prior_var_new, rho_new_data, q_new_data, rho_new_fn, q_new_fn, X_new = self.sample_split_or_keep_going_posterior('keep_going',node_idx,keep_going,Y_total,v,X,X_total,VAR,VAR_total,v_prior_mat,rho_data,q_data,rho_fn,q_fn)
                #print(accept)
                #if node_idx == 2:
                #    print(self.nodes[node_idx].keep_going)
                if accept:
                    # update tree
                    self.nodes = new_tree

                    # update matrices
                    rho_data = rho_new_data
                    q_data = q_new_data

                    rho_fn = rho_new_fn
                    q_fn = q_new_fn

                    X = X_new
                    X_total = torch.cat([X, v_prior_mat])
                    VAR_total = torch.cat([VAR, v_prior_var_new])

                    keep_going = keep_going_new

            v = self.sample_value_posterior_GIBBS(X_total, Y_total, VAR_total)
            for i in range(0, self.num_nodes):
                self.nodes[i].value = v[i]

        #return torch.matmul(X_total,v)

    def sample_value_posterior_GIBBS(self,X,Y,V):
        # with the basic priors, the conditional posterior for the values reduces
        # to a Bayesian linear regression
        sigma = torch.cholesky_inverse(torch.cholesky(torch.matmul(torch.matmul(X.T,torch.diag(1/V)),X)))
        mu = torch.matmul(sigma,torch.matmul(torch.matmul(X.T,torch.diag(1/V)),Y))

        return torch.distributions.MultivariateNormal(mu,sigma).sample()

    def sample_split_or_keep_going_posterior(self,sample_type,node_idx,keep_going,Y_total,v,X,X_total_old,VAR,VAR_total_old,v_prior_mat,rho_data,q_data,rho_fn,q_fn):
        # sample type is a string, either 'split' or 'keep_going'

        if not ((sample_type=='split') or (sample_type=='keep_going')):
            raise ValueError("sample_type must be split or keep_going")

        # copy of tree for MCMC moves
        new_tree = list()
        for i in range(0, self.num_nodes):
            new_tree.append(self.nodes[i].clone_self())

        keep_going_new = keep_going.detach().clone()
        node = new_tree[node_idx]

        # proposal for split is a Beta distribution
        if sample_type == 'split':
            node.split_proportion = torch.distributions.Beta(node.split_proportion*self.split_proposal_tune,(1-node.split_proportion)*self.split_proposal_tune).sample()

        # proposal for the keep_going underlying variable is a Normal distribution
        if sample_type == 'keep_going':
            #node.keep_going_logistic = torch.distributions.Normal(node.keep_going_logistic,self.keep_going_logistic_proposal_tune).sample()
            node.keep_going_logistic = sample_StandardLogistic(node.keep_going_prob.shape)
            alpha = node.keep_going_prob / (1 - node.keep_going_prob)
            node.keep_going =  (1.0 / (1.0 + torch.exp(-(torch.log(alpha) + node.keep_going_logistic) / self.concrete_temp)))

        # !!Everything below can be made much more efficient by only doing calculations for the node
        # !!and its descendents. We're just calculating everything right now to avoid bugs.

        for i in range(0,self.num_nodes):
            keep_going_new[i] = new_tree[i].keep_going

        # calculate intervals
        for i in range(0, self.num_nodes):
            new_tree[i].calc_child_intervals(new_tree)

        # copy of rho & q for MCMC moves
        rho_new_data = None
        q_new_data = None
        if (self.data is not None):
            rho_new_data = rho_data.detach().clone()
            q_new_data = q_data.detach().clone()

        rho_new_fn = None
        q_new_fn = None
        if self.regularizer_function is not None:
            rho_new_fn = rho_fn.detach().clone()
            q_new_fn = q_fn.detach().clone()

        v_prior_var_new = torch.zeros(self.num_nodes)
        interval_size = self.nodes[0].right_point - self.nodes[0].left_point
        v_prior_var_new[0] = math.pow(self.sigma_v0, 2.0)
        for i in range(1,self.num_nodes):
            v_prior_var_new[i] = math.pow(self.nodes[i].calc_value_std(self.nodes, self.mrange, self.sigma_va, self.sigma_vb),2.0)

        # calculate probabilities down the tree
        for i in range(0, self.num_nodes):
            node = new_tree[i]
            if node.is_leaf is False:
                if self.regularizer_function is not None:
                    rho_new_fn[:, i] = 1.0 / (1.0 + torch.exp(-self.alpha_rho * (self.regularizer_function.inducing_points - node.split_point)))
                    q_new_fn[:, node.left()] = (1 - rho_new_fn[:, i]) * q_new_fn[:, i] * node.keep_going
                    q_new_fn[:, node.right()] = rho_new_fn[:, i] * q_new_fn[:, i] * node.keep_going

                if (self.data is not None):
                    rho_new_data[:, i] = 1.0 / (1.0 + torch.exp(-self.alpha_rho * (self.data - node.split_point)))
                    q_new_data[:, node.left()] = (1 - rho_new_data[:, i]) * q_new_data[:, i] * node.keep_going
                    q_new_data[:, node.right()] = rho_new_data[:, i] * q_new_data[:, i] * node.keep_going

        if self.data is None or self.regularizer_function is None:
            if (self.data is not None):
                X_new = q_new_data*(1.0-keep_going_new)
            else:
                X_new = q_new_fn * (1.0 - keep_going_new)
        else:
            X_new = torch.cat([q_new_data*(1.0-keep_going_new), q_new_fn * (1.0 - keep_going_new)])

        X_total_new = torch.cat([X_new, v_prior_mat])
        VAR_total_new = torch.cat([VAR, v_prior_var_new])

        # compute posterior probability

        new_log_prob = torch.sum(torch.distributions.Normal(torch.matmul(X_total_new,v),torch.sqrt(VAR_total_new)).log_prob(Y_total))
        #print("new_log_prob= " + str(new_log_prob))
        if sample_type == 'split':
            # !!the prior on the split term is currently uniform, so doesn't enter the calculation
            new_log_prior_prob = 0.0
            log_prob_prop_old_given_new = torch.distributions.Beta(new_tree[node_idx].split_proportion*self.split_proposal_tune,(1-new_tree[node_idx].split_proportion)*self.split_proposal_tune).log_prob(self.nodes[node_idx].split_proportion)

        if sample_type == 'keep_going':
            new_log_prior_prob = logpdf_StandardLogistic(new_tree[node_idx].keep_going_logistic)
            # proposal distribution is symmetric, so doesn't enter into calculation
            log_prob_prop_old_given_new = 0.0

        ll_new = new_log_prob + log_prob_prop_old_given_new + new_log_prior_prob

        old_log_prob = torch.sum(torch.distributions.Normal(torch.matmul(X_total_old, v), torch.sqrt(VAR_total_old)).log_prob(Y_total))
        #print("old_log_prob= " + str(old_log_prob))

        if sample_type == 'split':
            log_prob_prop_new_given_old = torch.distributions.Beta(self.nodes[node_idx].split_proportion * self.split_proposal_tune,(1 - self.nodes[node_idx].split_proportion) * self.split_proposal_tune).log_prob(new_tree[node_idx].split_proportion)
            old_log_prior_prob = 0.0

        if sample_type == 'keep_going':
            old_log_prior_prob = logpdf_StandardLogistic(self.nodes[node_idx].keep_going_logistic)
            log_prob_prop_new_given_old = 0.0

        ll_old = old_log_prob + log_prob_prop_new_given_old + old_log_prior_prob

        #if node_idx == 2 and sample_type == 'keep_going':
        #    print("new_log_prob=",new_log_prob)
        #    print("old_log_prob=",old_log_prob)

        u = torch.log(torch.rand(1))
        #print("Accept prob. = ",torch.exp(ll_new-ll_old))
        #print("Old split = ",self.nodes[node_idx].split_proportion)
        #print("New split = ", new_tree[node_idx].split_proportion)

        accept = False
        if u < (ll_new - ll_old):
            # accept the move
            accept = True

        return accept,new_tree,keep_going_new,v_prior_var_new,rho_new_data,q_new_data,rho_new_fn,q_new_fn,X_new

x = torch.linspace(50,500,50)
linfn = LinearRegularizer(x,1.0)
fn_val = linfn.eval()
tree = UnivariateRegressionTree(3,[50,500],linfn)
tree.sample_splits_from_prior()
tree.calc_intervals()
tree.sample_values_from_prior()
tree.visualize_tree('prior_tree.png')

v = tree.eval_tree(x)
fig, ax = plt.subplots()
#ax.plot(x,v)

for i in range(0,500):
    if i % 20 == 0:
        print(i)
    tree.sample_posterior(1)
    v2 = tree.eval_tree(x)
    ax.plot(x,v2)
plt.show()
tree.visualize_tree('posterior_tree.png')