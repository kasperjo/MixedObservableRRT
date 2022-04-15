import matplotlib.pyplot as plt
import numpy as np



def flatten_list(lst):
    """
    Flattens a nested array and returns an array of all objects in 'lst' that are not lists
    """
    flattened = []

    def flatten_list_helper(lst_new):
        for item in lst_new:
            if type(item) == list:
                flatten_list_helper(item)
            else:
                flattened.append(item)

    flatten_list_helper(lst)
    return flattened

def plot(model, path, color, label=None, star=False, linewidth=6):
    """
    :param model: model of Class Model
    :param path: Node path
    :param color: Color of path
    :param label: Label of plotted path
    """
    tree = model.root
    all_x_vals = [[] for _ in range(len(tree.xy_cords))]
    all_y_vals = [[] for _ in range(len(tree.xy_cords))]
    for node in path:
        for i in range(len(tree.xy_cords)):
            all_x_vals[i].append(node.state[tree.xy_cords[i][0]])
            all_y_vals[i].append(node.state[tree.xy_cords[i][1]])
            if node.observed:
                plt.scatter(node.state[tree.xy_cords[i][0]], node.state[tree.xy_cords[i][1]], marker='o', color='k', s=150, zorder=1)
    for i in range(len(tree.xy_cords)):
        plt.plot(all_x_vals[i], all_y_vals[i], '--', c=color, label=label, linewidth=linewidth, zorder=-1)
        if star:
            plt.scatter(all_x_vals[i][-1], all_y_vals[i][-1], marker='*', c='purple', s=150, zorder=1)

    plt.xlim(tree.Xi[0])
    plt.ylim(tree.Xi[1])

def plot_all_observations(model, tree=None):
    """
    Plots all observation nodes in model, starting at model.root and ending at tree
    """
    if tree is None:
        tree = model.root
    for node in tree.all_nodes:
        if node.observed:
            plt.plot(node.state[tree.xy_cords[0][0]], node.state[tree.xy_cords[0][1]], 'o', color='k')
    for child in tree.children:
        plot_all_observations(child)


def return_subpath(node, obs_number):
    """
    Returns subpath for the end_node 'node', and hierarchy_number obs_number
    """
    node_temp = node.copy()
    RRT = node_temp.RRT
    while RRT.hierarchy_number != obs_number:
        node_temp = RRT.start
        RRT = RRT.parent

    end_node = node_temp
    start_node = RRT.start

    path = [end_node]
    curr_node = end_node.parent
    while curr_node != start_node.parent:
        path.append(curr_node)
        curr_node = curr_node.parent
    path.reverse()

    return path


def plot_path(path, style, color):
    """
    :param path: path consisting of nodes
    :param style: plot style, e.g., '--' or ':'
    :param color: color of path in plot
    """
    x_vals = []
    y_vals = []
    for node in path:
        x_vals.append(node.state[0])
        y_vals.append(node.state[1])
    plt.plot(x_vals, y_vals, style, color=color)
    plt.plot(x_vals[-1], y_vals[-1], '*', color='purple')


def plot_environment(model):
    """
    Draw the model environment in a plot. Aka, draw obstacle, observation areas, and hidden goal states
    """
    model.root.draw_region(model.root.obstacles)
    model.root.draw_region(model.root.observation_areas)

    for i, goal_state in enumerate(model.root.goal_states):
        plt.plot(goal_state[0], goal_state[1], 'o', color='r')
        plt.annotate('e=' + str(i), (goal_state[0] + 0.3, goal_state[1] + 0.3))


def run_MORRT(model):
    """
    Returns Mixed Observable RRT model as well as the best plan from the model
    """

    # Generate plans
    model.get_plan()

    # Find the best plan
    best_plan = model.best_plan()

    return model, best_plan

def get_heading(start_node, end_node):
    x1 = start_node.state[0]
    x2 = end_node.state[0]
    y1 = start_node.state[1]
    y2 = end_node.state[1]

    slope = (y2 - y1) / (x2 - x1)

    if (slope < 0) and (y2 < y1):
        return -np.arctan(np.abs(slope))
    elif (slope < 0) and (y2 > y1):
        return np.pi - np.arctan(np.abs(slope))
    elif (y1 > y2) and (x1 > x2):  # This implies a positive slope, but we "want it to be negative"
        return np.pi + np.arctan(slope)
    else:
        return np.arctan(slope)

    # node_index = path.index(node)

    # if node_index == len(path)-1:
    #     if node_index == 0:  # If there is only one node in path
    #         return 0
    #     elif node_index == 1: # If there are two nodes in path and node is second
    #         parent_node = path[node_index-1]
    #         x1 = parent_node.state[0]
    #         y1 = parent_node.state[1]
    #         x2 = node.state[0]
    #         y2 = node.state[1]
    #         slope = (y2-y1) / (x2-x1)
    #         return np.arctan(slope)
    #     else:
    #         parent_node = path[node_index - 1]
    #         grandparent_node = path[node_index - 2]
    #         xvals = np.array([grandparent_node.state[0], parent_node.state[0], node.state[0]]).reshape(3,)
    #         yvals = np.array([grandparent_node.state[1], parent_node.state[1], node.state[1]]).reshape(3,)
    #         f = interp1d(xvals, yvals, kind=2, fill_value="extrapolate")
    #         slope = derivative(f, node.state[0])
    #         return np.arctan(slope)
    # else:
    #     if len(path) == 2:  # If there are two nodes in path and node is first
    #         child_node = path[node_index+1]
    #         x1 = node.state[0]
    #         y1 = node.state[1]
    #         x2 = child_node.state[0]
    #         y2 = child_node.state[1]
    #         slope = (y2 - y1) / (x2 - x1)
    #         return np.arctan(slope)
    #     else:
    #         parent_node = path[node_index - 1]
    #         child_node = path[node_index + 1]
    #         xvals = np.array([parent_node.state[0], node.state[0], child_node.state[0]]).reshape(3, )
    #         yvals = np.array([parent_node.state[1], node.state[1], child_node.state[1]]).reshape(3, )
    #         f = interp1d(xvals, yvals, kind=2, fill_value="extrapolate")
    #
    #         slope = derivative(f, node.state[0])
    #
    #         return np.arctan(slope)


def _get_heading(node, path):
    node_index = path.index(node)
    if node_index == 0:  # If there is only one node in path
        if len(path) == 1:
            return 0
    else:
        child_node = path[0]
        x1 = node.state[0]
        y1 = node.state[1]
        x2 = child_node.state[0]
        y2 = child_node.state[1]
        slope = (y2 - y1) / (x2 - x1)
        print('slope', slope)
        print('heading', np.arctan(slope))
        if (slope < 0) and (y2 < y1):
            return np.arctan(slope)
            # return -np.arctan(np.abs(slope))
        elif (slope < 0) and (y2 > y1):
            return np.arctan(slope)
            # return np.pi - np.arctan(np.abs(slope))
        elif (y1 > y2) and (x1 > x2):  # This implies a positive slope, but we "want it to be negative"
            return np.arctan(slope)
            # return np.pi + np.arctan(slope)
        else:
            return np.arctan(slope)


def get_goal(path):
    """
    returns next goal node for MPC, given a "sub-path" to next observation
    """
    index = 0
    start_node = path[index]
    end_node = path[index]
    while not start_node.RRT.obstacle_between(start_node, path[index + 1]):
        index += 1
        end_node = path[index]
        if index == len(path) - 1:
            break
    plt.plot(end_node.state[0], end_node.state[1], 'o', color='k')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    print('index', index)
    return end_node


def get_plan_node():
    """
    Returns desired heading angle at goal node
    """
    pass

def path_to_array(path):
    """
    :param path: path of nodes
    :return: path in the form of a numpy array with states 
    """
    arr = []
    for node in path:
        arr.append(node.state)
    arr = np.array(arr)
    print(arr)

    return arr

def array_to_path(arr, tree):
    """
    :param arr: path in the form of a numpy array with states
    :return: path of nodes
    """
    from Version12_generalizedPlan_oneAgent import Node
    arr = arr.tolist()
    path = []
    for coords in arr:
        node_temp = Node(np.array(coords))
        node_temp.RRT = tree
        path.append(node_temp)

    return path

def return_nodes_to_follow(subpath):
    """
    Returns nodes for hardware/sim to follow
    """
    indx = 0
    start = subpath[0]
    path = [start]
    end = subpath[0]

    if len(subpath) == 1:
        finished = True
    else:
        finished = False
    while not finished:
        indx += 1
        end = subpath[indx]
        if indx == len(subpath)-1:
            path.append(end)
            finished = True
        elif start.RRT.obstacle_between(start, subpath[indx+1]): 
            path.append(end)
            start = end

    return path


