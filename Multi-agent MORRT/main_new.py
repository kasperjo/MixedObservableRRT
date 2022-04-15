from Version13_dynamicProgramming_multiAgent import *
from helper_functions import *

save_plans = False
np.random.seed(0)

# # =================================================================
### Root RRT ###

# Parameters
x0_1 = Node(np.array([0, 0]))
x0_2 = Node(np.array([0, 0]))
starts = [x0_1, x0_2]
Xi = [[0, 5], [0, 5]]
Delta = 0.25
Q = 0.5 * Delta * 1e2 * np.eye(2)  # Stage cost
QN = 1e4 * 0.5 * Delta * np.eye(2)  # Terminal cost
xg0 = np.array([0.5, 4.5])
xg1 = np.array([4.5, 4.5])
goal_states = [xg0, xg1]
gamma = 10000  # RRT* radius parameter
eta = 6 * Delta  # RRT* radius parameter
Theta1 = np.array([[0.8, 0], [0, 0.2]])
Theta2 = np.array([[0.2, 0], [0, 0.8]])
Omega = np.eye(2)
b0 = np.array([0.5, 0.5])
N = 500
N_subtrees = 5

observation_area1 = ObservationArea([[4, 5], [4, 5]], [Theta1, Theta2])
observation_area2 = ObservationArea([[0, 1], [4, 5]], [Theta1, Theta2])

# Define agent parameters
obstacles_agent_1 = [[[2.5, 3], [2, 5]]]  # Actual obstacles
obstacles_agent_1_plan = [[[2.5, 3], [2, 5]]]  # Obstacles to plan around
obstacles_agent_2 = None

obstacles = {0: obstacles_agent_1_plan, 1: obstacles_agent_2}

observation_areas_agent_1 = [observation_area1, observation_area2]
observation_areas_agent_2 = [observation_area1, observation_area2]

all_observation_areas = [observation_area1, observation_area2]
all_obstacles = obstacles_agent_1

observation_areas = {0: observation_areas_agent_1, 1: observation_areas_agent_2}

# Create the root RRT
RRT_root = RRT(starts=starts, Xi=Xi, Delta=Delta, Q=Q, QN=QN, goal_states=goal_states, Omega=Omega, v0=b0, star=True,
               gamma=gamma, eta=eta,
               obstacles=obstacles,
               observation_areas=observation_areas, N_subtrees=N_subtrees)

### Mixed Observable RRT model ###
model = Model(RRT_root, N)

# # =================================================================
# Uncomment the following to run and plot

model, best_plan = run_MORRT(model)
# plot_environment(model)
plan_ends = flatten_plan(best_plan)

### Plot environment and closed loop trajectories
plt.figure()
plot_agent_plans(model, best_plan)

# Observation Areas
for i, observation_area in enumerate(all_observation_areas):
    x_min, x_max = observation_area.region[0][0], observation_area.region[0][1]
    y_min, y_max = observation_area.region[1][0], observation_area.region[1][1]
    if i == 0:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5,
                                  label='Observation Areas', zorder=-1)
    else:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, zorder=-1)
    plt.gca().add_patch(rectangle)

# Obstacles
for i, obstacle in enumerate(all_obstacles):
    x_min, x_max = obstacle[0][0], obstacle[0][1]
    y_min, y_max = obstacle[1][0], obstacle[1][1]
    if i == 0:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
    else:
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k")
    plt.gca().add_patch(rectangle)

# Goal Regions
plt.scatter(xg0[0], xg0[1], color='blue', edgecolors='k', label='Goal for $e=0$', s=200, zorder=1)
plt.scatter(xg1[0], xg1[1], color='red', edgecolors='k', label='Goal for $e=1$', s=200, zorder=1)
# # plt.plot(0, 10, 'o', color='r', label='e=3')
# plt.annotate('e=0', (xg0[0]+0.3, xg0[1]+0.3))
# plt.annotate('e=1', (xg1[0]+0.3, xg1[1]+0.3))
plt.xlim(Xi[0])
plt.ylim(Xi[1])

handles, labels = plt.gca().get_legend_handles_labels()
#plt.gca().legend(handles[2:4] + handles[0:2], labels[2:4] + labels[0:2], loc=4)
# plt.legend()
plt.show()

## Save plans as numpy arrays
if save_plans:
    # Agent 1
    path_e1_Agent1 = return_nodes_to_follow(return_subpath(plan_ends[0][0], 0, agent=0),
                                            agent=0) + return_nodes_to_follow(
        return_subpath(plan_ends[0][0], 1, agent=0), agent=0)
    path_e2_Agent1 = return_nodes_to_follow(return_subpath(plan_ends[0][1], 0, agent=0),
                                            agent=0) + return_nodes_to_follow(
        return_subpath(plan_ends[0][1], 1, agent=0), agent=0)
    path_e1_Agent1 = path_to_array(path_e1_Agent1)
    path_e2_Agent1 = path_to_array(path_e2_Agent1)

    pathForObs_e1 = open("path_e1_Agent1.txt", "w")
    for row in path_e1_Agent1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e1, row, delimiter=',')
    pathForObs_e1.close()

    pathForObs_e2 = open("path_e2_Agent1.txt", "w")
    for row in path_e2_Agent1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e2, row, delimiter=',')
    pathForObs_e2.close()

    # Agent 2
    path_e1_Agent2 = return_nodes_to_follow(return_subpath(plan_ends[1][0], 0, agent=1),
                                            agent=1) + return_nodes_to_follow(
        return_subpath(plan_ends[1][0], 1, agent=1), agent=1)
    path_e2_Agent2 = return_nodes_to_follow(return_subpath(plan_ends[1][1], 0, agent=1),
                                            agent=1) + return_nodes_to_follow(
        return_subpath(plan_ends[1][1], 1, agent=1), agent=1)
    path_e1_Agent2 = path_to_array(path_e1_Agent2)
    path_e2_Agent2 = path_to_array(path_e2_Agent2)

    pathForObs_e1 = open("path_e1_Agent2.txt", "w")
    for row in path_e1_Agent2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e1, row, delimiter=',')
    pathForObs_e1.close()

    pathForObs_e2 = open("path_e2_Agent2.txt", "w")
    for row in path_e2_Agent2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e2, row, delimiter=',')
    pathForObs_e2.close()
