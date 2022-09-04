import argparse

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace_dir', type=str, help='test workspaces are taken from that dir '
                                                        'splited workspaces are saved there')
    parser.add_argument('--easy_ratio', type=float, default=50, help='ratio of easy workspaces')
    parser.add_argument('--medium_ratio', type=float, default=40, help='ratio of medium workspaces')
    parser.add_argument('--hard_ratio', type=float, default=10, help='ratio of hard workspaces')

    args = parser.parse_args()

    easy_med_hard_ratio = [args.easy_ratio, args.medium_ratio, args.hard_ratio]

    trajectories = np.load(args.workspace_dir + '/trajectories_test.npz')
    trajectories = [trajectories[str(i)] for i in range(len(trajectories))]
    trajectories_len = [len(trajectories[i]) for i in range(len(trajectories))]

    workspaces_indices_sorted = np.argsort(trajectories_len)
    split_indices = len(trajectories) * np.cumsum(easy_med_hard_ratio) / 100

    workspaces = np.load(args.workspace_dir + '/test_workspaces.npy')

    easy_workspaces = workspaces[workspaces_indices_sorted[:int(split_indices[0])]]
    medium_workspaces = workspaces[workspaces_indices_sorted[int(split_indices[0]):int(split_indices[1])]]
    hard_workspaces = workspaces[workspaces_indices_sorted[int(split_indices[1]):]]

    # save and plot:
    np.save(args.workspace_dir + '/test_workspaces_easy.npy', easy_workspaces)
    np.save(args.workspace_dir + '/test_workspaces_medium.npy', medium_workspaces)
    np.save(args.workspace_dir + '/test_workspaces_hard.npy', hard_workspaces)

    trajectories_len = np.array(trajectories_len)

    plt.hist(trajectories_len[workspaces_indices_sorted[int(split_indices[1]):]])
    plt.title('Hard Trajectories Length Histogram')
    plt.show()

    plt.hist(trajectories_len[workspaces_indices_sorted[int(split_indices[0]):int(split_indices[1])]])
    plt.title('Medium Trajectories Length Histogram')
    plt.show()

    plt.hist(trajectories_len[workspaces_indices_sorted[:int(split_indices[0])]])
    plt.title('Easy Trajectories Length Histogram')
    plt.show()


