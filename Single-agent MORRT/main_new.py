from Version12_generalizedPlan_oneAgent import *
from helper_functions import *
from matplotlib import pyplot as plt

save_plan = False
np.random.seed(0)

# # =================================================================
### Root RRT ###

# Parameters
x0 = Node(np.array([2.5, 0.8]))  # Start point
Xi = [[0, 5], [0, 5]]  # Constraint set
Delta = 0.1  # Incremental distance in RRT
Q = 0.5 * Delta * 1e2 * np.eye(2)  # Stage cost
QN = 1e4 * 0.5 * Delta * np.eye(2)  # Terminal cost
xg0 = np.array([0.5, 4.5])  # First partially observable goal state
xg1 = np.array([4.5, 4.5])  # Second partially observable goal state
goal_states = [xg0, xg1]
gamma = 7500  # RRT* radius parameter
eta = 6 * Delta  # RRT* radius parameter
Theta1 = np.array([[0.8, 0], [0, 0.2]])  # Observation accuracy matrix
Theta2 = np.array([[0.2, 0], [0, 0.8]])  # Observation accuracy matrix
Omega = np.eye(2)  # Partially observable environment transition matrix
b0 = np.array([1 / 2, 1 / 2])  # Initial belief
obstacles = [[[0.5, 4.5], [1.33, 1.66]]]  # Obstacles
obstacles_plan = [[[0.5, 4.5], [1.33, 1.66]]]  # Obstacles for plan (larger than actual obstacles)
observation_area1 = ObservationArea([[0, 5], [0, 0.5]], [Theta1, Theta2])  # First observation area
observation_area2 = ObservationArea([[0, 5], [2.16, 2.66]], [Theta1, Theta2])  # Second observation area
observation_areas = [observation_area1, observation_area2]  # TODO: Add observation_area2 for experiment
N = 500  # Number of nodes for final RRT
N_child = 5  # Number of children of each RRT
star = True  # RRT-star if True

# Create the root RRT
RRT_root = RRT(start=x0, Xi=Xi, Delta=Delta, Q=Q, QN=QN, goal_states=goal_states, Omega=Omega, v0=b0,
          star=star, gamma=gamma, eta=eta, obstacles=obstacles_plan, observation_areas=observation_areas,
          N_subtrees=N_child)

### Mixed Observable RRT model ###
model = Model(RRT_root, N)



# # =================================================================
# Uncomment the following to run and plot

model, best_plan = run_MORRT(model)
# plot_environment(model)
plan_ends = flatten_list(best_plan)



# x_cl_nlp_obs1 = np.loadtxt('observation1.txt')
# x_cl_nlp_obs2 = np.loadtxt('observation2.txt')

### Plot environment and plan
plt.figure()
model.plot_plan(plan_ends)


# Goal Regions
plt.scatter(xg0[0], xg0[1], color='blue', edgecolors='k', label = 'Goal for $e=0$', s=65, zorder=1)
plt.scatter(xg1[0], xg1[1], color='red', edgecolors='k', label = 'Goal for $e=1$', s=65, zorder=1)
# # plt.plot(0, 10, 'o', color='r', label='e=3')
# plt.annotate('e=0', (xg0[0]+0.3, xg0[1]+0.3))
# plt.annotate('e=1', (xg1[0]+0.3, xg1[1]+0.3))
plt.xlim(Xi[0])
plt.ylim(Xi[1])

# Observation Area
for i, observation_area in enumerate(observation_areas):
    x_min, x_max = observation_area.region[0][0], observation_area.region[0][1]
    y_min, y_max = observation_area.region[1][0], observation_area.region[1][1]
    if i == 0:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, label='Observation Areas', zorder=-1)
    else:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, zorder=-1)
    plt.gca().add_patch(rectangle)

# Obstacles
for i, obstacle in enumerate(obstacles):
    x_min, x_max = obstacles[0][0][0], obstacles[0][0][1]
    y_min, y_max = obstacles[0][1][0], obstacles[0][1][1]
    if i == 0:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
    else:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
    plt.gca().add_patch(rectangle)

handles, labels = plt.gca().get_legend_handles_labels()
#plt.gca().legend(handles[2:4] + handles[0:2], labels[2:4] + labels[0:2])

# plt.legend()
plt.show()

## Save plans as numpy arrays
if save_plan:

    path_obs1 = return_nodes_to_follow(return_subpath(plan_ends[0], 0)) + return_nodes_to_follow(return_subpath(plan_ends[0], 1))
    path_obs2 = return_nodes_to_follow(return_subpath(plan_ends[1], 0)) + return_nodes_to_follow(return_subpath(plan_ends[1], 1))

    path_obs1 = path_to_array(path_obs1)
    path_obs2 = path_to_array(path_obs2)

    pathForObs1 = open("pathForObs1.txt", "w")
    for row in path_obs1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs1, row, delimiter=',')
    pathForObs1.close()

    pathForObs2 = open("pathForObs2.txt", "w")
    for row in path_obs2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs2, row, delimiter=',')
    pathForObs2.close()

