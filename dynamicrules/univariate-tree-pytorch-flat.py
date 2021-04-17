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


# helper functions to compute neighbors in the tree
def left_node(idx):
    return 2*idx + 1

def right_node(idx):
    return 2*idx + 2

def parent_node(idx):
    return int((idx-1)/2)

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

        # tree variables and parameters
        self.depths = torch.zeros(self.num_nodes,dtype=torch.int)
        self.is_leaf = list()
        self.keep_going_probs = torch.ones(self.num_nodes)
        self.keep_going = torch.zeros(self.num_nodes)
        self.keep_going_logistic = torch.zeros(self.num_nodes)
        self.values = torch.zeros(self.num_nodes)
        self.left_points = torch.zeros(self.num_nodes)
        self.right_points = torch.zeros(self.num_nodes)
        self.split_proportions = torch.zeros(self.num_nodes)
        self.split_points = torch.zeros(self.num_nodes)
        self.clades = list()

        # temporary tree variables used for MCMC
        self.new_keep_going = torch.zeros(self.num_nodes)
        self.new_keep_going_logistic = torch.zeros(self.num_nodes)
        self.new_left_points = torch.zeros(self.num_nodes)
        self.new_right_points = torch.zeros(self.num_nodes)
        self.new_split_proportions = torch.zeros(self.num_nodes)
        self.new_split_points = torch.zeros(self.num_nodes)
        self.rho_data = None
        self.rho_data_new = None
        self.q_data = None
        self.q_data_new = None
        self.rho_fn = None
        self.rho_fn_new = None
        self.q_fn = None
        self.q_fn_new = None
        self.v_prior_mat = None
        self.v_prior_var = None
        self.v_prior_var_new = None

        self.X_data = None
        self.X_fn = None
        self.Y_hat_fn = None
        self.Y_hat_data = None

        self.y_fn = None
        self.var_fn = None

        if self.data is not None:
            # 'soft decisions' at cut points
            self.rho_data = torch.zeros((self.data.size, self.num_nodes))
            # probability of reaching node
            self.q_data = torch.ones((self.data.size, self.num_nodes))
            self.var_data = torch.ones(self.data.size) * math.pow(self.sigma_data, 2.0)

        if self.regularizer_function is not None:
            # 'soft decisions' at cut points
            self.rho_fn = torch.zeros((len(self.regularizer_function.inducing_points), self.num_nodes))
            # probability of reaching node
            self.q_fn = torch.ones((len(self.regularizer_function.inducing_points), self.num_nodes))
            self.y_fn = self.regularizer_function.eval()
            self.var_fn = torch.ones(len(self.regularizer_function.inducing_points)) * math.pow(self.sigma_data,2.0) / (self.zeta * self.function_reg_constant)

        self.v_prior_mat = torch.zeros((self.num_nodes, self.num_nodes))
        self.v_prior_var = torch.zeros(self.num_nodes)
        self.v_prior_var_new = self.v_prior_var.detach().clone()
        self.y_v_prior = torch.zeros(self.num_nodes)

        # prior on first value
        self.v_prior_mat[0, 0] = 1.0
        self.v_prior_var[0] = math.pow(self.sigma_v0, 2.0)
        self.y_v_prior[0] = self.mu_v0
        # prior on parent and child
        for i in range(1, self.num_nodes):
            parentidx = parent_node(i)
            self.v_prior_mat[i, parentidx] = -1.0
            self.v_prior_mat[i, i] = 1.0

        # compute tree node properties
        for i in range(0,self.num_nodes):
            if left_node(i) >= math.pow(2, depth + 1) - 1:
                self.is_leaf.append(True)
            else:
                self.is_leaf.append(False)
                self.depths[left_node(i)] = self.depths[i] + 1
                self.depths[right_node(i)] = self.depths[i] + 1
            self.keep_going_probs[i] = self.mlambda1 * torch.pow(1 + self.depths[i], -self.mlambda2)

        for i in range(0,self.num_nodes):
            clade = list()
            self.clade_recurse(i,clade)
            clade.sort()
            self.clades.append(torch.tensor(clade,dtype=torch.int))

        self.left_points[0] = torch.tensor(interval[0])
        self.right_points[0] = torch.tensor(interval[1])

    def clade_recurse(self,idx,myclade):
        # recursively compute clade (node + all its descendents)
        myclade.append(idx)
        if self.is_leaf[idx]:
            return
        self.clade_recurse(left_node(idx),myclade)
        self.clade_recurse(right_node(idx), myclade)

    def calc_child_intervals(self,idx,mleft_points,mright_points,msplit_points,msplit_proportions):
        # no children, nothing to do
        if self.is_leaf[idx]:
            return

        delta = mright_points[idx]-mleft_points[idx]
        msplit_points[idx] = msplit_proportions[idx]*delta + mleft_points[idx]

        mleft_points[left_node(idx)] = mleft_points[idx]
        mright_points[left_node(idx)] = msplit_points[idx]

        mright_points[right_node(idx)] = mright_points[idx]
        mleft_points[right_node(idx)] = msplit_points[idx]

    def calc_value_std(self,idx,mleft_points,mright_points):
        parentidx = parent_node(idx)
        midpoint = (mright_points[idx] + mleft_points[idx]) / 2.0
        parent_midpoint = (mright_points[parentidx] + mleft_points[parentidx]) / 2.0
        return (self.sigma_va + self.sigma_vb * torch.abs(parent_midpoint - midpoint) / self.mrange)

    def print_node_interval(self,idx):
        return "[" + str(self.left_points[idx]) + ", " + str(self.right_points[idx]) + "]"

    def sample_split_from_prior(self,idx):
        # no split at leaf
        if self.is_leaf[idx]:
            return

        self.split_proportions[idx] = torch.distributions.Beta(1.0, 1.0).sample()

    def sample_keep_going_from_prior(self,idx):
        # always stop if a leaf
        if self.is_leaf[idx]:
            return
        self.keep_going_logistic[idx], self.keep_going[idx] = sample_BinaryConcrete(self.keep_going_probs[idx], self.concrete_temp)

    def sample_values_from_prior(self):
        # sample initial value
        self.values[0] = torch.distributions.Normal(self.mu_v0,self.sigma_v0).sample()

        # sample keep_going values down the tree
        for i in range(0,self.num_nodes):
            self.sample_keep_going_from_prior(i)

        # sample values down the tree
        for i in range(1,self.num_nodes):
            parentidx = parent_node(i)
            #interval_proportion = (self.nodes[i].right_point-self.nodes[i].left_point)/self.range
            self.values[i] = self.values[parentidx] + torch.distributions.Normal(torch.tensor(0.0),self.calc_value_std(i)).sample()
        self.keep_going[0] = torch.tensor(1.0)

    def sample_splits_from_prior(self):
        for i in range(0, self.num_nodes):
            self.sample_split_from_prior(i)

    def calc_intervals(self):
        for i in range(0, self.num_nodes):
            self.calc_child_intervals(i)

    def eval_tree(self,x):
        # x is the vector of values to evaluate the tree

        # in more efficient implementation, could pre-allocate and re-use arrays

        # 'soft decisions' at cut points
        rho = torch.zeros((len(x),self.num_nodes))

        # probability of reaching node
        q = torch.ones((len(x),self.num_nodes))

        self.calc_tree_probs_below(x, rho, q, 0)

        return torch.matmul(q*(1.0-self.keep_going), self.values)

    def calc_tree_probs_below(self,x,rho,q,start_idx,msplit_points,mkeep_going):
        # x is the vector of values to evaluate the tree
        # rho is a # nodes X data size matrix for storing 'soft decisions' at cut points
        # q is a # nodes X data size matrix for storing probabilities of reaching nodes
        # start_idx is node index to start evaluation

        # push probabilities down the tree
        for i in range(start_idx,self.num_nodes):
            if self.is_leaf[i] is False:
                rho[:,i] = 1.0/(1.0+torch.exp(-self.alpha_rho*(x-msplit_points[i])))
                q[:,left_node(i)] = (1-rho[:,i])*q[:,i]*mkeep_going[i]
                q[:,right_node(i)] = rho[:,i] * q[:,i] * mkeep_going[i]

    def visualize_tree(self,fname):
        graph = pydot.Dot(graph_type='graph')
        self.visualize_tree_recurse(0,graph)
        #print(graph.to_string())
        graph.write_png(fname)

    def visualize_tree_recurse(self,idx,graph):
        #viznode = pydot.Node(str(node.idx),label=node.print_interval()+" val=" + str(node.value),shape='box')
        if self.keep_going[idx] <= 0.95:
            interval = '[' + f'{self.left_points[idx]:.2f}' + ', ' + f'{self.right_points[idx]:.2f}' + ']'
            viznode = pydot.Node(str(idx),label=interval + '\n' + "val=" + f'{self.values[idx]:.2f}',shape='box',color='red')
        else:
            #viznode = pydot.Node(str(node.idx),label=' ',shape='box', color='blue')
            viznode = pydot.Node(str(idx), label="val=" + f'{self.values[idx]:.2f}', shape='box', color='blue')
        graph.add_node(viznode)
        if self.keep_going[idx] > 0.95:
            edge_left = pydot.Edge(str(idx),str(left_node(idx)),label='<'+f'{self.split_points[idx]:.2f}')
            graph.add_edge(edge_left)
            edge_right = pydot.Edge(str(idx),str(right_node(idx)))
            graph.add_edge(edge_right)
            self.visualize_tree_recurse(left_node(idx),graph)
            self.visualize_tree_recurse(right_node(idx),graph)

    def init_sampling_matrices(self):
        if self.data is not None:
            self.calc_tree_probs_below(self.data, self.rho_data, self.q_data, 0, self.split_points, self.keep_going)

        if self.regularizer_function is not None:
            self.calc_tree_probs_below(self.regularizer_function.inducing_points, self.rho_fn, self.q_fn, 0, self.split_points, self.keep_going)

        # prior on adjacent values
        for i in range(1,self.num_nodes):
            parentidx = parent_node(i)
            self.v_prior_var[i] = math.pow(self.calc_value_std(i,self.left_points,self.right_points),2.0)

        if (self.data is not None):
            self.X_data = self.q_data * (1.0 - self.keep_going)

        if (self.regularizer_function is not None):
            self.X_fn = self.q_fn * (1.0 - self.keep_going)

    def sample_posterior(self,num_iters):
        self.init_sampling_matrices()

        # sample values
        self.sample_values_posterior_GIBBS()

        # sample splits
        for node_idx in range(0,self.num_nodes):
            if self.is_leaf[node_idx] is False:
                accept = self.sample_split_or_keep_going_posterior('split',node_idx)
                self.sample_values_posterior_GIBBS()

        # sample keep_going
        for node_idx in range(0,self.num_nodes):
            if self.is_leaf[node_idx] is False:
                accept = self.sample_split_or_keep_going_posterior('keep_going',node_idx)
                self.sample_values_posterior_GIBBS()

        #return torch.matmul(X_total,v)

    def sample_values_posterior_GIBBS(self):
        # with the basic priors, the conditional posterior for the values reduces
        # to a Bayesian linear regression
        if self.data is None or self.regularizer_function is None:
            if (self.data is not None):
                Y = self.data
                VAR = self.var_data
                X = self.X_data
            else:
                Y = self.y_fn
                VAR = self.var_fn
                X = self.X_fn
        else:
            Y = torch.cat([self.data, self.y_fn])
            VAR = torch.cat([self.var_data, self.var_fn])
            X = torch.cat([self.X_data, self.X_fn])

        Y_total = torch.cat([Y, self.y_v_prior])
        VAR_total = torch.cat([VAR, self.v_prior_var])
        X_total = torch.cat([X, self.v_prior_mat])

        sigma = torch.cholesky_inverse(torch.cholesky(torch.matmul(torch.matmul(X_total.T,torch.diag(1/VAR_total)),X_total)))
        mu = torch.matmul(sigma,torch.matmul(torch.matmul(X_total.T,torch.diag(1/VAR_total)),Y_total))

        self.values = torch.distributions.MultivariateNormal(mu,sigma).sample()

        if (self.data is not None):
            self.Y_hat_data = torch.matmul(self.X_data, self.values)

        if (self.regularizer_function is not None):
            self.Y_hat_fn = torch.matmul(self.X_fn, self.values)

    def sample_split_or_keep_going_posterior(self,sample_type,idx):
        # sample type is a string, either 'split' or 'keep_going'

        if not ((sample_type=='split') or (sample_type=='keep_going')):
            raise ValueError("sample_type must be split or keep_going")

        # proposal for split is a Beta distribution
        if sample_type == 'split':
            self.new_split_proportions[idx] = torch.distributions.Beta(self.split_proportions[idx]*self.split_proposal_tune,(1-self.split_proportions[idx])*self.split_proposal_tune).sample()
            for m in self.clades[idx]:
                self.calc_child_intervals(m,self.new_left_points,self.new_right_points,self.new_split_points,self.new_split_proportions)
            for m in self.clades[idx]:
                self.v_prior_var_new[m] = math.pow(self.calc_value_std(m,self.new_left_points,self.new_right_points),2.0)

        # proposal for the keep_going underlying variable is a Normal distribution
        ## right now sampling from the prior - fix this later
        if sample_type == 'keep_going':
            #node.keep_going_logistic = torch.distributions.Normal(node.keep_going_logistic,self.keep_going_logistic_proposal_tune).sample()
            self.new_keep_going_logistic[idx] = sample_StandardLogistic(self.keep_going_probs[idx].shape)
            alpha = self.new_keep_going_probs[idx] / (1 - self.new_keep_going_probs[idx])
            self.new_keep_going[idx] =  (1.0 / (1.0 + torch.exp(-(torch.log(alpha) + self.new_keep_going_logistic[idx]) / self.concrete_temp)))

        for m in self.clades[idx]:
            if self.is_leaf[m] is False:
                if self.regularizer_function is not None:
                    self.calc_tree_probs_below(self.regularizer_function.inducing_points, self.new_rho_fn, self.new_q_fn, m, self.new_split_points, self.new_keep_going)
                if self.data is not None:
                    self.calc_tree_probs_below(self.data, self.new_rho_data, self.new_q_data, m, self.new_split_points, self.new_keep_going)

        if self.regularizer_function is not None:
            X_new_fn_slice = self.new_q_fn[:,self.clades[idx]]*(1-self.new_keep_going[self.clades[idx]])
            Y_hat_fn_slice_new = torch.matmul(X_new_fn_slice,self.values[self.clades[idx]])
            Y_hat_fn_slice_remove = torch.matmul(self.X_fn[:,self.clades[idx]], self.values[self.clades[idx]])
        if self.data is not None:
            X_new_data_slice = self.new_q_data[:,self.clades[idx]]*(1-self.new_keep_going[self.clades[idx]])
            Y_hat_data_slice_new = torch.matmul(X_new_data_slice, self.values[self.clades[idx]])
            Y_hat_data_slice_remove = torch.matmul(self.X_data[:, self.clades[idx]], self.values[self.clades[idx]])

        # compute posterior probability
        new_log_prob = 0.0
        if self.regularizer_function is not None:
            Y_hat_fn_new = self.Y_hat_fn - Y_hat_fn_slice_remove + Y_hat_fn_slice_new
            new_log_prob += torch.sum(torch.distributions.Normal(Y_hat_fn_new,self.var_fn)).log_prob(self.regularizer_function.inducing_points)
        if self.data is not None:
            Y_hat_data_new = self.Y_hat_data - Y_hat_data_slice_remove + Y_hat_data_slice_new
            new_log_prob += torch.sum(torch.distributions.Normal(Y_hat_data_new,self.var_data)).log_prob(self.data)
        if idx > 0:
            parentidx = parent_node(idx)
            new_log_prob += torch.distributions.Normal(self.values[parentidx],self.v_prior_var_new[idx]).log_prob(self.values[idx])
        if self.is_leaf[idx] is False:
            leftidx = left_node(idx)
            rightidx = right_node(idx)
            new_log_prob += torch.distributions.Normal(self.values[idx], self.v_prior_var_new[leftidx]).log_prob(self.values[leftidx])
            new_log_prob += torch.distributions.Normal(self.values[idx], self.v_prior_var_new[rightidx]).log_prob(self.values[rightidx])

        if sample_type == 'split':
            # !!the prior on the split term is currently uniform, so doesn't enter the calculation
            new_log_prior_prob = 0.0
            log_prob_prop_old_given_new = torch.distributions.Beta(self.new_split_proportions[idx]*self.split_proposal_tune,(1-self.split_proportions[idx])*self.split_proposal_tune).log_prob(self.split_proportions[idx])

        if sample_type == 'keep_going':
            new_log_prior_prob = logpdf_StandardLogistic(self.new_keep_going_logistic[idx])
            # proposal distribution is symmetric, so doesn't enter into calculation
            log_prob_prop_old_given_new = 0.0

        ll_new = new_log_prob + log_prob_prop_old_given_new + new_log_prior_prob

        old_log_prob = 0.0
        if self.regularizer_function is not None:
            old_log_prob += torch.sum(torch.distributions.Normal(self.Y_hat_fn)).log_prob(self.regularizer_function.inducing_points)
        if self.data is not None:
            old_log_prob += torch.sum(torch.distributions.Normal(self.Y_hat_data, self.var_data)).log_prob(self.data)
        if idx > 0:
            parentidx = parent_node(idx)
            old_log_prob += torch.distributions.Normal(self.values[parentidx], self.v_prior_var[idx]).log_prob(self.values[idx])
        if self.is_leaf[idx] is False:
            leftidx = left_node(idx)
            rightidx = right_node(idx)
            old_log_prob += torch.distributions.Normal(self.values[idx], self.v_prior_var[leftidx]).log_prob(self.values[leftidx])
            old_log_prob += torch.distributions.Normal(self.values[idx], self.v_prior_var[rightidx]).log_prob(self.values[rightidx])

        if sample_type == 'split':
            log_prob_prop_new_given_old = torch.distributions.Beta(self.split_proportions[idx] * self.split_proposal_tune,(1 - self.split_proportions[idx]) * self.split_proposal_tune).log_prob(self.new_split_proportions[idx])
            old_log_prior_prob = 0.0

        if sample_type == 'keep_going':
            old_log_prior_prob = logpdf_StandardLogistic(self.keep_going_logistic[idx])
            log_prob_prop_new_given_old = 0.0

        ll_old = old_log_prob + log_prob_prop_new_given_old + old_log_prior_prob
        u = torch.log(torch.rand(1))

        if u < (ll_new - ll_old):
            # accept the move
            accept = True
            ## copy all the values from new to current
            if sample_type == 'split':
                self.left_points[self.clades[idx]] = self.new_left_points[self.clades[idx]]
                self.right_points[self.clades[idx]] = self.new_right_points[self.clades[idx]]
                self.split_points[self.clades[idx]] = self.new_split_points[self.clades[idx]]
                self.split_proportions[self.clades[idx]] = self.new_split_proportions[self.clades[idx]]
                self.v_prior_var[self.clades[idx]] = self.v_prior_var_new[self.clades[idx]]
            if sample_type == 'keep_going':
                self.keep_going[self.clades[idx]] = self.new_keep_going[self.clades[idx]]
                self.keep_going_logistic[self.clades[idx]] = self.new_keep_going_logistic[self.clades[idx]]
            if self.regularizer_function is not None:
                self.Y_hat_fn = Y_hat_fn_new
                self.rho_fn[:,self.clades[idx]] = self.rho_fn_new[:,self.clades[idx]]
                self.q_fn[:, self.clades[idx]] = self.q_fn_new[:, self.clades[idx]]
            if self.data is not None:
                self.Y_hat_data = Y_hat_data_new
                self.rho_data[:, self.clades[idx]] = self.rho_data_new[:, self.clades[idx]]
                self.q_data[:, self.clades[idx]] = self.q_data_new[:, self.clades[idx]]
        else:
            accept = False
            ## to copy all values from current to new (overwriting rejected new values)
            if sample_type == 'split':
                self.new_left_points[self.clades[idx]] = self.left_points[self.clades[idx]]
                self.new_right_points[self.clades[idx]] = self.right_points[self.clades[idx]]
                self.new_split_points[self.clades[idx]] = self.split_points[self.clades[idx]]
                self.new_split_proportions[self.clades[idx]] = self.split_proportions[self.clades[idx]]
                self.v_prior_var_new[self.clades[idx]] = self.v_prior_var[self.clades[idx]]
            if sample_type == 'keep_going':
                self.new_keep_going[self.clades[idx]] = self.keep_going[self.clades[idx]]
                self.new_keep_going_logistic[self.clades[idx]] = self.keep_going_logistic[self.clades[idx]]
            if self.regularizer_function is not None:
                Y_hat_fn_new = self.Y_hat_fn
                self.rho_fn_new[:,self.clades[idx]] = self.rho_fn[:,self.clades[idx]]
                self.q_fn_new[:, self.clades[idx]] = self.q_fn[:, self.clades[idx]]
            if self.data is not None:
                Y_hat_data_new = self.Y_hat_data
                self.rho_data_new[:, self.clades[idx]] = self.rho_data[:, self.clades[idx]]
                self.q_data_new[:, self.clades[idx]] = self.q_data[:, self.clades[idx]]

        return accept

x = torch.linspace(50,500,50)
linfn = LinearRegularizer(x,1.0)
fn_val = linfn.eval()
tree = UnivariateRegressionTree(3,[50,500],linfn)
tree.sample_splits_from_prior()
tree.calc_intervals()
tree.sample_values_from_prior()
tree.visualize_tree('prior_tree.png')

tree.init_sampling_matrices()
tree.sample_values_posterior_GIBBS()
tree.visualize_tree('posterior_tree.png')

v = tree.eval_tree(x)
fig, ax = plt.subplots()
#ax.plot(x,v)

#for i in range(0,500):
#    if i % 20 == 0:
#        print(i)
#    tree.sample_posterior(1)
#    v2 = tree.eval_tree(x)
#    ax.plot(x,v2)
#plt.show()
#tree.visualize_tree('posterior_tree.png')