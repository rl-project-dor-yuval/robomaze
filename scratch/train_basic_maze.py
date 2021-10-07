import MazeEnv as mz
from MazeEnv import Rewards
import time
import numpy as np
from stable_baselines3 import DDPG
from Training.Evaluation import EvalAndSaveCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

START_LOC = (5, 3.2)
TIMEOUT_STEPS = 200
BUFFER_SIZE = 1000  # smaller buffer for small task
TOTAL_TIME_STEPS = 10000
LEARNING_RATE = 0.001

REWARDS = Rewards(target_arrival=1, collision=-1, timeout=-0.5)

EVAL_EPISODES = 1
EVAL_FREQ = 100
VIDEO_FREQ = 4

# HER parameters
N_SAMPLED = 3
STRATEGY = 'future'  # futute, random or episode
ONLINE_SAMPLING = True


# plot_results()
# load_results()

def make_circular_map(size, radius):
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map


if __name__ == "__main__":
    start = time.time()

    # create environment :
    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    maze_map = make_circular_map(map_size, 3 / tile_size)
    # maze_map = np.zeros(map_size)

    maze_env = Monitor(mz.MazeEnv(maze_size=maze_size,
                                  maze_map=maze_map,
                                  tile_size=tile_size,
                                  start_loc=START_LOC,
                                  target_loc=np.divide(maze_size, 2),
                                  timeout_steps=TIMEOUT_STEPS,
                                  show_gui=False,
                                  rewards=REWARDS),
                       filename="logs/DummyMaze/results")
    _ = maze_env.reset()

    check_env(maze_env)

    # create separete evaluation environment:
    eval_maze_env = Monitor(mz.MazeEnv(maze_size=maze_size,
                                       maze_map=maze_map,
                                       tile_size=tile_size,
                                       start_loc=START_LOC,
                                       target_loc=np.divide(maze_size, 2),
                                       timeout_steps=TIMEOUT_STEPS,
                                       show_gui=False,
                                       rewards=REWARDS)
                            )
    _ = eval_maze_env.reset()

    # create model:
    model = DDPG(policy="MultiInputPolicy",
                 env=maze_env,
                 buffer_size=BUFFER_SIZE,
                 learning_rate=LEARNING_RATE,
                 device='cuda',
                 train_freq=(1, "episode"),
                 #              replay_buffer_class=HerReplayBuffer,
                 #              replay_buffer_kwargs=dict(
                 #                  n_sampled_goal=N_SAMPLED,
                 #                  goal_selection_strategy=STRATEGY,
                 #                  online_sampling=ONLINE_SAMPLING,
                 #                  max_episode_length=TIMEOUT_STEPS,
                 #              ),
                 verbose=1)

    # create callback for evaluation
    callback = EvalAndSaveCallback(log_dir="../Training/logs/DummyMaze",
                                   eval_env=eval_maze_env,
                                   eval_freq=EVAL_FREQ,
                                   eval_episodes=EVAL_EPISODES,
                                   eval_video_freq=VIDEO_FREQ,
                                   verbose=1)

    model.learn(total_timesteps=TOTAL_TIME_STEPS,
                callback=callback)

    print("time", time.time() - start)

    # model.save("models/basic_maze")

    # maze_env.reset()  # has to be called to save video
