import cv2
import numpy as np
import colorsys
import seaborn as sns


num_of_workspaces = 100
min_num_hard_workspaces = 30
# minimum number of samples that are guaranteed to have start and goal in different
# sides which makes the agent walk through the bottleneck to succeed
min_distance = 15 # in maze pixels, for all targets


def generate_start_and_goal(points):
    while True:
        start_idx = np.random.choice(points.shape[0], 1)
        goal_idx = np.random.choice(points.shape[0], 1)
        start = np.squeeze(points[start_idx, :])
        goal = np.squeeze(points[goal_idx, :])
        if np.linalg.norm(start - goal) > min_distance:
            return start, goal


free_space_map = - (cv2.imread("maps/bottleneck_freespace.png", cv2.IMREAD_GRAYSCALE) / 255) + 1

free_space_points = np.argwhere(free_space_map == 0)

workspaces = []
# generate hard workspaces:
for i in range(min_num_hard_workspaces):
    while True:
        start, goal = generate_start_and_goal(free_space_points)
        if (start[1] < 30 and goal[1] > 60) or (start[1] > 60 and goal[1] < 30):
            # they are on different sides of the map
            workspaces.append((start, goal))
            break

# generate the rest of random workspaces
for i in range(num_of_workspaces - min_num_hard_workspaces):
    workspaces.append(generate_start_and_goal(free_space_points))


# create plot for each workspace and save
for i, w in enumerate(workspaces):
    w_im = np.copy(-free_space_map + 1)
    cv2.circle(w_im, tuple(reversed(w[0])), 1, 0.2, -1)
    cv2.circle(w_im, tuple(reversed(w[1])), 2, 0.4, -1)

    cv2.imwrite(f"workspaces/bottleneck_plots/{i}.png", w_im * 255)


# create plot of all workspaces
colors = sns.color_palette(None, num_of_workspaces)
color_freespace_map = cv2.imread("maps/bottleneck_freespace.png", cv2.IMREAD_COLOR) / 255

for i, w in enumerate(workspaces):
     color_freespace_map[tuple(w[0])] = color_freespace_map[tuple(w[1])] = colors[i]
cv2.imwrite(f"workspaces/bottleneck_plots/all.png", color_freespace_map * 255)


# save workspaces to csv:
workspaces = np.array(workspaces)
np.save("workspaces/bottleneck", workspaces)
