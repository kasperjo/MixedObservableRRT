import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.special as func
import cProfile
import itertools
import time
from helper_functions import *

a = time.time()

np.random.seed(0)


#### TODO: Fix stop observation after perfect observation (as in muli-agent code)


class Model:
    def __init__(self, root, N_iters):
        """
        :param root: The initial Rapidly-Exploring Randon Tree. From Class RRT
        :param N_iters: Number of nodes to add for final RRT, when there are no more observations to be made
        """
        self.root = root
        self.N_iters = N_iters
        self.all_RRTs = [self.root]
        self.N_subtrees = self.root.N_subtrees

    def best_plan(self, tree=None):
        """
        This function returns the best plan using a depth first search and dynamical programming approach.
        The best plan is returned as a nested array with end nodes.
        self.get_plan() must be run first in order to generate the possible plans.
        """
        if tree is None:
            tree = self.root

        if not tree.children:
            end_nodes, costs = tree.return_end_nodes(only_lowest_costs=True)

            if tree == self.root:
                return end_nodes
            else:
                # sum the costs for each plan, and pair end nodes for each plan
                costs_temp = []
                plans = []
                for i in range(tree.N_goal_states ** (tree.hierarchy_number - 1)):
                    cost_temp = costs[tree.N_goal_states * i:tree.N_goal_states * (i + 1)]
                    cost_temp_sum = np.sum(cost_temp)
                    costs_temp.append(cost_temp_sum)

                    plan_temp = end_nodes[tree.N_goal_states * i:tree.N_goal_states * (i + 1)]
                    plans.append(plan_temp)

                costs = costs_temp

                return plans, costs

        else:  # tree has children
            child_plans, child_costs = [], []
            for child in tree.children:
                child_plan, child_cost = self.best_plan(child)
                child_plans.append(child_plan)
                child_costs.append(child_cost)

            if tree == self.root:
                no_obs_end, no_obs_cost = tree.return_end_nodes(only_lowest_costs=True)

                child_cost = np.inf
                child_plan = None

                for i in range(len(child_plans)):
                    child = tree.children[i]
                    plan = child_plans[i]
                    cost = child_costs[i]

                    obs_node = child.start
                    cost_to_obs = obs_node.parent.path_costs + obs_node.parent.node_costs

                    cost = np.array(cost) + np.array(cost_to_obs)

                    if np.sum(cost) < np.sum(child_cost):  # Sum to remove list items
                        child_cost = cost
                        child_plan = plan

                if np.sum(child_cost) < np.sum(no_obs_cost):
                    return child_plan

                else:
                    return no_obs_end

            else:  # Tree is not root
                best_ends_no_obs, costs_no_obs = tree.return_end_nodes(only_lowest_costs=True)

                final_plan, final_costs = [[] for _ in range(tree.N_goal_states ** (tree.hierarchy_number - 1))], [[]
                                                                                                                   for _
                                                                                                                   in
                                                                                                                   range(
                                                                                                                       tree.N_goal_states ** (
                                                                                                                               tree.hierarchy_number - 1))]

                for i in range(tree.N_goal_states ** (tree.hierarchy_number - 1)):
                    for j in range(tree.N_goal_states):

                        end_no_obs, cost_no_obs = best_ends_no_obs[i + j], costs_no_obs[i + j]

                        child_cost = np.inf
                        child_plan = None

                        for k in range(len(child_plans)):
                            child = tree.children[k]
                            plan = child_plans[k][i + j]
                            cost = child_costs[k][i + j]

                            obs_node = child.start
                            cost_to_obs = obs_node.parent.path_costs[0][i + j] + obs_node.parent.node_costs[0][
                                i + j]  # For observation j. TODO: Is this right????

                            cost = np.array(cost) + np.array(cost_to_obs)

                            if cost < child_cost:
                                child_cost = cost
                                child_plan = plan

                        if child_cost < cost_no_obs:
                            final_costs[i].append(child_cost)
                            final_plan[i].append(child_plan)
                        else:
                            final_costs[i].append(cost_no_obs)
                            final_plan[i].append(end_no_obs)

                ## "Merge" final_costs
                costs_temp = []

                for costs in final_costs:
                    costs_temp.append(np.sum(costs))

                final_costs = costs_temp

                return final_plan, final_costs

    def plot_plan(self, end_nodes, colors=None):
        """
        Plots the possible paths in a plan, given all possible end nodes in array end_nodes

        :param end_nodes: list of all possible end nodes in a plan
        """
        ## The environment is drawn from separately using helper_functions atm
        # self.root.draw_region(self.root.obstacles)
        # self.root.draw_region(self.root.observation_areas)

        paths = []

        for end_node in end_nodes:
            paths.append(self.root.return_path(end_node))

        for i, path in enumerate(paths):
            if colors is None:
                plot(self, path, 'r', star=True)
            else:
                plot(self, path, 'r', star=True)

        plt.xlim(self.root.Xi[0])
        plt.ylim(self.root.Xi[1])

    def build_tree(self, tree):
        """
        get_plan() helper function
        """
        # if len(tree.observed_areas) == len(tree.observation_areas):  # No more observations can be made

        if len(tree.observation_areas) == len(tree.observed_areas):
            for _ in range(self.N_iters):
                tree.add_node()
        else:  # Grow tree until enough observations have been made
            while len(tree.observations) < tree.N_subtrees:
                tree.add_node()

    def get_plan(self, tree=None):
        """
        Creates a sample of plans
        """
        if tree is None:
            tree = self.root
        self.build_tree(tree)
        self.get_child_plan(tree)

    def get_child_plan(self, tree):
        """
        Helper function of get_plan()
        """
        print('Number of observations', len(tree.observations))
        # number_of_sub_RRTs = min(len(tree.observations), self.N_subtrees)

        number_of_sub_RRTs = len(tree.observations)

        print(self.N_subtrees)
        for n in trange(number_of_sub_RRTs):
            observation = tree.observations[n]
            node_obs = observation[0]
            area_obs = observation[1]
            area_index = observation[2]

            # Only count the cost from observation
            node_obs.path_costs = np.zeros(node_obs.path_costs.shape)
            node_obs.path_length = 0

            RRT_temp = RRT(node_obs, tree.Xi, tree.Delta, tree.Q, tree.QN, tree.goal_states, tree.Omega,
                           node_obs.vs, star=tree.star, gamma=tree.gamma,
                           eta=tree.eta,
                           obstacles=tree.obstacles,
                           observation_areas=tree.observation_areas, N_subtrees=tree.N_subtrees)
            RRT_temp.observed_areas = tree.observed_areas.copy()
            RRT_temp.observed_areas.append(area_obs)

            if tree.observation_areas[
                area_index].perfect_obs:  # If observation is perfect, no more observations are needed
                RRT_temp.observed_areas = tree.observation_areas

            # print(RRT_temp.observation_areas)
            # if RRT_temp.observation_areas:
            #     RRT_temp.observation_areas.remove(area_obs)

            # Set initialized to true, since we have vs
            RRT_temp.initialized = True

            RRT_temp.start_cost = node_obs.parent.path_costs + node_obs.parent.node_costs

            RRT_temp.obs_cost = node_obs.parent.path_costs

            RRT_temp.hierarchy_number = tree.hierarchy_number + 1
            self.get_plan(RRT_temp)
            RRT_temp.parent = tree
            self.all_RRTs.append(RRT_temp)
            tree.children.append(RRT_temp)


class Node:
    """Node of Mixed Observable Rapidly-Exploring Random Tree"""

    def __init__(self, state, parent=None, children=None, RRT=None, path_length=0):
        """
        :param state: state of the node
        :param parent: parent node
        :param children: array of child nodes
        :param RRT: the RRT that the node belongs to
        :param path_length: the length of a path starting at Node.RRT.start and ending at node
        """
        if children is None:
            children = []
        self.vs = []  # An array of all unnormalized belief vectors
        self.RRT = RRT
        self.dim = state.reshape(-1, ).shape[0]
        self.state = state.reshape(self.dim, 1)
        self.parent = parent
        self.children = children

        if not self.vs:
            self.path_costs = np.zeros((1, 1))
            self.node_costs = np.zeros((1, 1))
            self.terminal_costs = np.zeros((1, 1))
        else:
            self.path_costs = np.zeros((1, RRT.N_goal_states * len(self.vs)))
            self.node_costs = np.zeros((1, RRT.N_goal_states * len(self.vs)))
            self.terminal_costs = np.zeros(
                (1, RRT.N_goal_states * len(self.vs)))

        self.observed = False  # True if observation is made at node
        self.observation_node = None  # Keep track of which "Parent node" made the observation higher up in the tree

        self.path_length = path_length

    def copy(self):
        """
        Returns copy of node object
        """
        node_new = Node(self.state.copy(), self.parent, self.children.copy(), self.RRT, self.path_length)
        node_new.vs = self.vs.copy()
        node_new.RRT = self.RRT
        node_new.observed = self.observed
        node_new.observation_node = self.observation_node

        return node_new


class RRT:
    """Mixed Observable Rapidly-Exploring Random Tree (MORRT)"""

    def __init__(self, start, Xi, Delta, Q, QN, goal_states, Omega, v0, star=True, gamma=None,
                 eta=None, obstacles=None,
                 observation_areas=None, hierarchy_number=0, N_subtrees=1):
        """
        :param start: root of RRT. Belongs to class Node
        :param Xi: array on form [[x1_min, x1_max],...,[xn_min,xn_max]], defining state constraints of nodes in RRT
        :param Delta: incremental distance of RRT
        :param Q: quadratic state cost
        :param QN: quadratic terminal cost
        :param goal_states: list of possible partially observable goal states, e.g., [xg1, xg2]. Store the state of a goal as a numpy column vector
        :param Omega: transition probability matrix of partially obsetvable environment states
        :param v0: initial belief vector(s)
        :param star: use RRT* algorithm if star=True. Use standard RRT 
        :param gamma: parameter for RRT* radius. Only applicable if star=True
        :param eta: max radius of RRT* ball. Radius then shrinks as a function of gamma. Only applicable if star=True
        :param obstacles: obstacles for agent. On form = [[[x_min, x_max], [y_min], y_max]], ...] square obstacles in xy-space
        :param observation_areas: array of ObservationArea Class objects, with all areas where the agent can make observations
        :param hierarchy_number: the depth in the tree of RRTs where self is. E.g. if self the parent of self has a parent which is the root RRT, then hierarchy_number=2 (root->parent->self)
        :param N_subtrees: number of child RRTs to initialize from observation nodes. Aka number of observations to make in RRT before initializing a new one
        """
        self.start = start
        self.dim = start.state.reshape(-1, ).shape[0]
        self.Xi = Xi
        self.Delta = Delta
        self.all_nodes = [start]
        self.all_edges = []
        self.Q = Q
        self.QN = QN
        self.goal_states = []
        for g in goal_states:
            self.goal_states.append(g.reshape(self.dim, 1))  # store goal states as column vectors
        self.star = star
        self.gamma = gamma
        self.eta = eta
        self.Omega = Omega

        try:  # A single belief vector (prior to any observation)
            self.v0 = v0.reshape(v0.reshape(-1, ).shape[0], 1)  # Make v0 a column vector
        except:  # More than one belief vector (aftern an observation)
            vs_temp = []
            for v in v0:
                vs_temp.append(v.reshape(v.reshape(-1, ).shape[0], 1))  # Make v a column vector
            self.v0 = vs_temp

        self.initialized = False
        self.obs_made = 0
        self.obstacles = obstacles
        self.xy_cords = [[0, 1]]
        self.observation_areas = observation_areas
        self.observations = []  # On form [[observation_node, observation_area, area_index],...]
        self.start_cost = 0  # Initial cost, eg. for observation node
        self.obs_cost = 0  # The current "final cost" when an observation is made
        self.children = []  # Keep track of children (from observations)
        self.observed_areas = []  # Each child RRT can not observe an already observed area
        self.hierarchy_number = hierarchy_number
        self.parent = None
        self.N_subtrees = N_subtrees  # Keep track of number of allowed observations
        self.N_goal_states = len(self.goal_states)

        self.shortest_path = 0
        self.shortest_node = self.start

    def add_node(self):
        """
        Adds node to RRT
        """
        if not self.initialized:
            self.start.RRT = self
            self.start.cost = 0
            self.start.vs = [self.v0]
            self.initialized = True

        # Sample random node
        rand_node = self.get_rand_node()

        # find nearest node in current RRT
        parent = self.find_nearest_node(rand_node)
        # Do nothing more if obstacle is in the way
        if self.obstacle_between(rand_node, parent):
            return None

        # create new node an incremental distance self.Delta from parent
        new_node = self.generate_new_node(parent, rand_node)

        # Update costs at new node
        new_node.parent = parent  # Temporarily set parent in order for return_node_number to work
        area_index, observation, area = self.get_observation(new_node)
        new_node.path_costs, new_node.terminal_costs, new_node.node_costs = self.compute_costs(new_node, observation,
                                                                                               area)

        # RRT*
        if (self.star) and (observation is None):
            ## RRT-star
            neighbors = self.find_neighbors(new_node)

            # Find best parent neighbor
            cost = np.sum(new_node.path_costs)
            for neighbor in neighbors:
                cost_temp = neighbor.path_costs + neighbor.node_costs
                if (np.sum(cost_temp) < cost) and (not neighbor.observed) and (
                        not self.child_with_observation(neighbor)) and (not self.obstacle_between(new_node, neighbor)):
                    parent = neighbor
                    cost = np.sum(cost_temp)

            # Do nothing more if obstacle is in the way
            if self.obstacle_between(new_node, parent):
                return None

            # Update costs at new node
            new_node.parent = parent
            new_node.path_costs, new_node.terminal_costs, new_node.node_costs = self.compute_costs(new_node,
                                                                                                   observation,
                                                                                                   area)

        # Update hierarchy
        parent.children.append(new_node)
        self.all_nodes.append(new_node)
        self.all_edges.append([parent, new_node])
        new_node.path_length = new_node.parent.path_length + 1

        # RRT*
        if (self.star) and (observation is None):
            ## RRT-star
            for neighbor in neighbors:
                curr_costs = neighbor.path_costs
                new_costs_temp = new_node.path_costs + new_node.node_costs
                if (np.sum(new_costs_temp) < np.sum(curr_costs)) and (not neighbor.observed) and (
                        not neighbor.parent.observed) and (not self.child_with_observation(neighbor)) and (
                        not self.obstacle_between(new_node,
                                                  neighbor)):  # TODO: Do not alter observation node??? Could probably include, but must make sure to keep observation

                    # Change hierarchy
                    neighbor.parent.children.remove(neighbor)
                    self.all_edges.remove([neighbor.parent, neighbor])
                    neighbor.parent = new_node
                    self.all_edges.append([new_node, neighbor])
                    neighbor.path_costs, neighbor.terminal_costs, neighbor.node_costs = self.compute_costs(neighbor,
                                                                                                           None,
                                                                                                           None)
                    new_node.children.append(neighbor)
                    neighbor.path_length = neighbor.parent.path_length + 1

        # Update parameters if observation was made
        if observation is not None:
            self.obs_made += 1
            new_node.observed = True
            self.observations.append([new_node, area, area_index])

        # Update final parameters
        boolean, obs_node = self.observation_in_path(new_node)
        new_node.observation_node = obs_node
        new_node.RRT = self

    def get_rand_node(self):
        """
        Returns a random node sampled uniformly from the constraint set
        """

        x_new = np.zeros((self.dim, 1))
        for i in range(self.dim):
            x_min = self.Xi[i][0]
            x_max = self.Xi[i][1]
            x_new[i, 0] = np.random.uniform(low=x_min, high=x_max)
        rand_node = Node(x_new)

        return rand_node

    @staticmethod
    def child_with_observation(node):
        """
        Returns True if observation amongst children of node. Returns False otherwise
        """
        for child in node.children:
            if child.observed:
                return True
        return False

    def find_neighbors(self, node):
        """
        Returns an array of node neighbors for RRT* algorithm
        """
        neighbors = []
        n_nodes = len(self.all_nodes)
        for node_temp in self.all_nodes:
            Vd = np.pi ** (self.dim / 2) / func.gamma(self.dim / 2 + 1)
            radius = min((self.gamma / Vd * np.log(n_nodes) / n_nodes) ** (1 / self.dim), self.eta)
            if np.linalg.norm(node_temp.state - node.state) < radius:
                neighbors.append(node_temp)
        return neighbors

    def return_end_nodes(self, only_lowest_costs=False):
        """
        :param lowest_cost: if True, only return end_nodes with lowest cost of one of outfalls
        :return: end nodes of RRT. Aka, returns nodes that do not have any children
        """
        end_nodes = []
        best_ends = [None for _ in range(self.N_goal_states ** self.hierarchy_number)]
        lowest_costs = [np.inf for _ in range(self.N_goal_states ** self.hierarchy_number)]

        for node in self.all_nodes:
            if (not node.children) and (not self.observation_in_path(node)[0]):  # TODO: Remember to add to general case
                if only_lowest_costs:
                    for i in range(len(lowest_costs)):
                        cost_temp = node.path_costs[0, i] + node.terminal_costs[0, i]
                        if cost_temp < lowest_costs[i]:
                            lowest_costs[i] = cost_temp
                            best_ends[i] = node
                else:
                    end_nodes.append(node)
        if only_lowest_costs:
            return best_ends, lowest_costs
        else:
            return end_nodes, None

    def get_observation(self, node):
        """
        :return: index of observation area, True, observation area (if observation is made at node)
                 -1, None, None if no observation is made
        """
        observed, area = self.is_inside(node, self.observation_areas)
        for ind, area_temp in enumerate(self.observation_areas):
            if area_temp == area:
                area_index = ind
        if observed:
            if (not self.observation_in_path(node)[0]) and (
                    area not in self.observed_areas):
                for observation in self.observations:
                    node_temp = observation[0]
                    if np.linalg.norm(node.state - node_temp.state) < 0:  # TODO: What radius?
                        return -1, None, None
                print('Observation made')
                return area_index, True, area

            else:
                return -1, None, None
            # else:
            #     return k, None
        # Return uninformative observation
        else:
            return -1, None, None

    def observation_in_path(self, node):  # TODO: Edited fast, so double check function...
        """
        Returns True if there is an observation at a previous node in the path starting at self.start
        """
        # if self.start.observed:
        #     return False, self.start  # TODO: We need an observation node but return False in order to be able to make observations in next RRT
        node_temp = node
        while node_temp != self.start:
            if node_temp.observed:
                return True, node_temp
            node_temp = node_temp.parent
        return False, None

    def get_C(self, observation, area):
        """
        Helper function for updating unnormalized belief vectors
        """
        if observation:
            C = []
            for Theta in area.Thetas:
                C.append(Theta @ self.Omega)
        else:
            C = [self.Omega]
        return C

    def get_vs(self, node, C):
        """
        Returns unnormalized belief vectors
        """
        vs_parent = node.parent.vs
        vs = []
        for v in vs_parent:
            for c in C:
                vs.append(c @ v)
        return vs

    def compute_costs(self, node, observation=None, area=None):
        """Computes cost at node (not include terminal cost) cost_internal,
        as well as terminal cost cost_terminal
        """
        C = self.get_C(observation, area)
        node.vs = self.get_vs(node, C)

        # Compute node, and terminal costs
        h = []
        hN = []
        for i in range(self.N_goal_states):
            h.append(self.cost_h(node, self.goal_states[i]))
            hN.append(self.cost_hN(node, self.goal_states[i]))

        path_costs = node.parent.path_costs + node.parent.node_costs

        N_vs = len(node.vs)
        node_costs = []
        terminal_costs = []
        for i in range(N_vs):
            node_costs.append(np.dot(h, node.vs[i]))
            terminal_costs.append(np.dot(hN, node.vs[i]))
        node_costs = np.array(node_costs).reshape((1, N_vs))
        terminal_costs = np.array(terminal_costs).reshape((1, N_vs))

        if observation:
            path_costs_temp = np.zeros(node_costs.shape)
            for i in range(int(node_costs.shape[1] / self.N_goal_states)):
                for j in range(self.N_goal_states):
                    path_costs_temp[0, self.N_goal_states * i + j] = path_costs[0, i]
            path_costs = path_costs_temp

        return path_costs, terminal_costs, node_costs

    def find_nearest_node(self, rand_node):
        """
        Returns the RRT-node closest to the node rand_node
        """
        nearest = None
        distance = np.inf
        for node in self.all_nodes:
            dist_temp = np.linalg.norm(node.state - rand_node.state)
            if dist_temp < distance:
                nearest = node
                distance = dist_temp

        return nearest

    def is_inside(self, node, constraint):
        """
        :param constraint: obstacles or observation_areas
        :param node:
        :return: True if node is inside of constraint. Also returns the specific area which the node is inside
        """
        for cords in self.xy_cords:
            x = node.state[cords[0]]
            y = node.state[cords[1]]
            if constraint == self.obstacles:
                if not self.obstacles:
                    return False, None
                for area in self.obstacles:
                    if (area[0][0] <= x <= area[0][1]) and (area[1][0] <= y <= area[1][1]):
                        return True, area
            elif constraint == self.observation_areas:
                if not self.observation_areas:
                    return False, None
                if self.observation_areas is not None:
                    for observation_area in self.observation_areas:
                        if (observation_area.region[0][0] <= x <= observation_area.region[0][1]) and (
                                observation_area.region[1][0] <= y <= observation_area.region[1][1]):
                            return True, observation_area
        return False, None

    def obstacle_between(self, node1, node2):
        """
        Checks if there is an obstacle between node1 and node1. Returns True/False
        """
        if self.obstacles is None:
            return False
        if self.is_inside(node1, self.obstacles)[0] or self.is_inside(node2, self.obstacles)[0]:
            return True

        for cords in self.xy_cords:
            x1 = node1.state[cords[0]]
            y1 = node1.state[cords[1]]
            x2 = node2.state[cords[0]]
            y2 = node2.state[cords[1]]
            p1 = Point(x1, y1)
            q1 = Point(x2, y2)
            for obstacle in self.obstacles:
                x_min = obstacle[0][0]
                x_max = obstacle[0][1]
                y_min = obstacle[1][0]
                y_max = obstacle[1][1]
                p2 = Point(x_min, y_min)
                q2 = Point(x_min, y_max)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_min, y_max)
                q2 = Point(x_max, y_max)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_max, y_max)
                q2 = Point(x_max, y_min)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_max, y_min)
                q2 = Point(x_min, y_min)
                if doIntersect(p1, q1, p2, q2):
                    return True
            return False

    def generate_new_node(self, parent, rand_node):
        """
        Creates new RRT node
        """
        dist = np.linalg.norm(parent.state - rand_node.state)
        if dist < self.Delta:  # In case rand_node is very close to parent
            new_state = rand_node.state
        else:
            new_state = parent.state + (rand_node.state - parent.state) / dist * self.Delta
        new_node = Node(new_state)
        return new_node

    def draw_tree(self):
        """
        Draws the RRT in a plot

        xy_cords consists of all cartesian coordinates from the state vector x
        They are ordered: [[x0, y0], [x1, y1],...]
        For example: [[0,3], [4,5]] implies the first and fourth elements of Node.state
        is [x0,y0], while the fifth and sixth element is [x1,y1] (zero indexed)
        """
        for edge in self.all_edges:
            parent, child = edge
            for cords in self.xy_cords:
                plt.plot([parent.state[cords[0]], child.state[cords[0]]],
                         [parent.state[cords[1]], child.state[cords[1]]], c='b')
        plt.xlim(self.Xi[0])
        plt.ylim(self.Xi[1])

    def draw_path(self, path):
        """
        :param path: an array of nodes, forming a path

        xy_cords consists of all cartesian coordinates from the state vector x
        They are ordered: [[x0, y0], [x1, y1],...]
        For example: [[0,3], [4,5]] implies the first and fourth elements of Node.state
        is [x0,y0], while the fifth and sixth element is [x1,y1] (zero indexed)
        """
        all_x_vals = [[] for _ in range(len(self.xy_cords))]
        all_y_vals = [[] for _ in range(len(self.xy_cords))]
        for node in path.ordered_nodes:
            for i in range(len(self.xy_cords)):
                all_x_vals[i].append(node.state[self.xy_cords[i][0]])
                all_y_vals[i].append(node.state[self.xy_cords[i][1]])
        for i in range(len(self.xy_cords)):
            plt.plot(all_x_vals[i], all_y_vals[i])

    def cost_h(self, node, xg):
        """
        Helper function for stage cost
        """
        h = (node.state - xg).T @ self.Q @ (node.state - xg)
        return float(h)

    def cost_hN(self, node, xg):
        """
        Helper function for terminal cost
        """
        hN = (node.state - xg).T @ self.QN @ (node.state - xg)
        return float(hN)

    @staticmethod
    def return_path(end_node):
        """
        :return: nodes ordered in path, starting at start of root RRT and ending at 'end_node'
        """
        path = [end_node]
        curr_node = end_node
        while curr_node.parent != None:
            curr_node = curr_node.parent
            path.append(curr_node)
        path.reverse()
        return path

    def draw_region(self, constraint):
        """
        Draws all regions in constraint set in a plot

        :param constraint: obstacles or observation_areas
        """
        if (constraint == self.obstacles) and (self.obstacles is not None):
            for area in self.obstacles:
                x_min, x_max = area[0][0], area[0][1]
                y_min, y_max = area[1][0], area[1][1]
                rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k")
                plt.gca().add_patch(rectangle)
        elif (constraint == self.observation_areas) and (self.observation_areas is not None):
            for observation_area in self.observation_areas:
                x_min, x_max = observation_area.region[0][0], observation_area.region[0][1]
                y_min, y_max = observation_area.region[1][0], observation_area.region[1][1]
                rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5)
                plt.gca().add_patch(rectangle)

        plt.xlim(self.Xi[0])
        plt.ylim(self.Xi[1])

    def return_node_number(self, node):
        """
        :param node: Tree node
        :return: The number 'k' where 'node' is the k:th node in path, i.e,
        the time step k, used in cost update equation
        """
        if node == self.start:
            return 0
        else:
            k = 1
            node_temp = node.parent
            while node_temp != self.start:
                k += 1
                node_temp = node_temp.parent
            return k


class ObservationArea:
    def __init__(self, region, Thetas):
        """
        :param region: Regions on form [[-x_min, x_max], [y_min, y_max]]
        :param Thetas: List of Thetas for region, [Theta1, Theta2,...], corresponding to noise in observing goal states
        """
        self.region = region
        self.Thetas = Thetas

        # See if observation is perfect
        perfect_theta = np.zeros(Thetas[0].shape)
        perfect_theta[0][0] = 1
        if (Thetas[0] == perfect_theta).all():
            self.perfect_obs = True
        else:
            self.perfect_obs = False


######### Code from Geeksforgeeks #######################------------------------------------------------------------
# A Python3 program to find if 2 given line segments intersect or not

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


# This code is contributed by Ansh Riyal
######### Code from Geeksforgeeks #######################------------------------------------------------------------


b = time.time()
